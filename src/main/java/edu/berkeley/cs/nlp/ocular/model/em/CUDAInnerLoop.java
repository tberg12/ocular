package edu.berkeley.cs.nlp.ocular.model.em;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunc_cache;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUsharedconfig;
import jcuda.driver.JCudaDriver;
import tberg.murphy.gpu.CudaUtil;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class CUDAInnerLoop implements EmissionCacheInnerLoop {
	
	public static final int BLOCK_SIZE_X = 16;
	public static final int ROLL_X = 17;
	public static final int BLOCK_SIZE_Y = 64;

	int numThreads;
	float[][] whiteTemplates;
	float[][] blackTemplates;
	int[] templateNumIndices;
	int[] templateIndicesOffsets;
	int maxTemplateWidth;
	int minTemplateWidth;
	int totalTemplateNumIndices;
	CUmodule cudaModule;
	CUdeviceptr d_Ow;
	CUdeviceptr d_Ob;
	CUdeviceptr d_scores;
	CUdeviceptr[] d_Tw;
	CUdeviceptr[] d_Tb;
	
	public CUDAInnerLoop(int numThreads, int gpuId) {
		this.numThreads = numThreads;
		
		CudaUtil.startup(gpuId);
		this.cudaModule = CudaUtil.compileAndLoad("/tmp/emission_cache_kernel", kernelSrcShared(), true);
//		this.cudaModule = CudaUtil.compileAndLoad("/tmp/emission_cache_kernel", kernelSrcPrivate(), true);
	}
	
	public void startup(float[][] whiteTemplates, float[][] blackTemplates, int[] templateNumIndices, int[] templateIndicesOffsets, int minTemplateWidth, int maxTemplateWidth, int maxSequenceLength, int totalTemplateNumIndices) {
		this.whiteTemplates = whiteTemplates;
		this.blackTemplates = blackTemplates;
		this.templateNumIndices = templateNumIndices;
		this.templateIndicesOffsets = templateIndicesOffsets;
		this.maxTemplateWidth = maxTemplateWidth;
		this.minTemplateWidth = minTemplateWidth;
		this.totalTemplateNumIndices = totalTemplateNumIndices;
		
		int numTemplateWidths = (maxTemplateWidth-minTemplateWidth)+1;
		int extendedMaxSeqLength = (BLOCK_SIZE_X*ROLL_X) * (int) Math.ceil(((double) maxSequenceLength) / (BLOCK_SIZE_X*ROLL_X));
		this.d_Ow = new CUdeviceptr();
		cuMemAlloc(d_Ow, (extendedMaxSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT * Sizeof.FLOAT);
		this.d_Ob = new CUdeviceptr();
		cuMemAlloc(d_Ob, (extendedMaxSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT * Sizeof.FLOAT);
		this.d_scores = new CUdeviceptr();
		cuMemAlloc(d_scores, maxSequenceLength*totalTemplateNumIndices * Sizeof.FLOAT);
		this.d_Tw = new CUdeviceptr[numTemplateWidths];
		this.d_Tb = new CUdeviceptr[numTemplateWidths];
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				d_Tw[tw-minTemplateWidth] = new CUdeviceptr();
				cuMemAlloc(d_Tw[tw-minTemplateWidth], whiteTemplates[tw-minTemplateWidth].length * Sizeof.FLOAT);
				cuMemcpyHtoD(d_Tw[tw-minTemplateWidth], Pointer.to(whiteTemplates[tw-minTemplateWidth]), whiteTemplates[tw-minTemplateWidth].length * Sizeof.FLOAT);
				
				d_Tb[tw-minTemplateWidth] = new CUdeviceptr();
				cuMemAlloc(d_Tb[tw-minTemplateWidth], blackTemplates[tw-minTemplateWidth].length * Sizeof.FLOAT);
				cuMemcpyHtoD(d_Tb[tw-minTemplateWidth], Pointer.to(blackTemplates[tw-minTemplateWidth]), blackTemplates[tw-minTemplateWidth].length * Sizeof.FLOAT);
			}
		}
	}

	public void shutdown() {
		cuMemFree(d_Ow);
		cuMemFree(d_Ob);
		cuMemFree(d_scores);
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				cuMemFree(d_Tw[tw-minTemplateWidth]);
				cuMemFree(d_Tb[tw-minTemplateWidth]);
			}
		}
	}

	public void compute(final float[] scores, final float[] whiteObservations, final float[] blackObservations, final int sequenceLength) {
		int gridSizeX = (int) Math.ceil(((double) sequenceLength) / (BLOCK_SIZE_X*ROLL_X));
		int extendedSeqLength = gridSizeX * (BLOCK_SIZE_X*ROLL_X);
		cuMemcpyHtoD(d_Ow, Pointer.to(CudaUtil.extendWithZeros(whiteObservations, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT)), (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT * Sizeof.FLOAT);
		cuMemcpyHtoD(d_Ob, Pointer.to(CudaUtil.extendWithZeros(blackObservations, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT)), (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT * Sizeof.FLOAT);
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				CUfunction function = new CUfunction();
				cuModuleGetFunction(function, cudaModule, "compute_emissions_"+tw);
				JCudaDriver.cuFuncSetCacheConfig(function, CUfunc_cache.CU_FUNC_CACHE_PREFER_SHARED);
				JCudaDriver.cuFuncSetSharedMemConfig(function, CUsharedconfig.CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE);
				Pointer kernelParameters = Pointer.to(Pointer.to(new int[] {templateIndicesOffsets[tw-minTemplateWidth]*sequenceLength}), Pointer.to(new int[] {sequenceLength}), Pointer.to(new int[] {templateNumIndices[tw-minTemplateWidth]}), Pointer.to(d_Tw[tw-minTemplateWidth]), Pointer.to(d_Tb[tw-minTemplateWidth]), Pointer.to(d_Ow), Pointer.to(d_Ob), Pointer.to(d_scores));
				int gridSizeY = (int) Math.ceil(((double) templateNumIndices[tw-minTemplateWidth]) / BLOCK_SIZE_Y);
				cuLaunchKernel(function, 
						gridSizeX, gridSizeY, 1,      // Grid dimension
						BLOCK_SIZE_X, BLOCK_SIZE_Y, 1,      // Block dimension
						0, null,               // Shared memory size and stream
						kernelParameters, null // Kernel- and extra parameters
						);
			}
		}
		cuMemcpyDtoH(Pointer.to(scores), d_scores, sequenceLength*totalTemplateNumIndices * Sizeof.FLOAT);
	}

	public int numOuterThreads() {
		return 1;
	}

	public int numPopulateThreads() {
		return numThreads;
	}
	
    public String kernelSrcShared() {
    	assert BLOCK_SIZE_X * BLOCK_SIZE_Y >= (BLOCK_SIZE_X*ROLL_X+(CharacterTemplate.LINE_HEIGHT-1));
    	StringBuffer buf = new StringBuffer();
    	for (int tw=1; tw<=CharacterTemplate.LINE_HEIGHT; ++tw) {
    		buf.append("extern \"C\"\n");
    		buf.append("__global__ void compute_emissions_"+tw+"(int scoresOffset, int Olength, int Tlength, float const* __restrict__ Tw, float const* __restrict__ Tb, float const* __restrict__ Ow, float const* __restrict__ Ob, float* scores) {\n");
    		
    		buf.append("__shared__ float sO["+(BLOCK_SIZE_X*ROLL_X+(tw-1))*CharacterTemplate.LINE_HEIGHT+"];\n");
    		buf.append("int sharedIndex = threadIdx.x * "+BLOCK_SIZE_Y+" + threadIdx.y;\n");
    		buf.append("if (sharedIndex < "+(BLOCK_SIZE_X*ROLL_X+(tw-1))+") {\n");
    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT+"; ++i) {\n");
    		buf.append("sO[sharedIndex * "+CharacterTemplate.LINE_HEIGHT+" + i] = Ow[(blockIdx.x * "+BLOCK_SIZE_X*ROLL_X+" + sharedIndex) * "+CharacterTemplate.LINE_HEIGHT+" + i];\n");
    		buf.append("}\n");
    		buf.append("}\n");
    		buf.append("__syncthreads();\n");

    		for (int r=0; r<ROLL_X; ++r) {
    			buf.append("float score"+r+" = 0;\n");
    		}
    		buf.append("int Tindex = blockIdx.y * "+BLOCK_SIZE_Y+" + threadIdx.y;\n");
    		buf.append("if (Tindex < Tlength) {\n");

    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT*tw+"; ++i) {\n");
    		buf.append("float tw = Tw[Tindex * "+CharacterTemplate.LINE_HEIGHT*tw+" + i];\n");
    		for (int r=0; r<ROLL_X; ++r) {
    			buf.append("score"+r+" = __fmaf_ru(sO[(threadIdx.x * "+ROLL_X+" + "+r+") * "+CharacterTemplate.LINE_HEIGHT+" + i], tw, score"+r+");\n");
    		}
    		buf.append("}\n");

    		buf.append("}\n");
    		buf.append("__syncthreads();\n");

    		buf.append("if (sharedIndex < "+(BLOCK_SIZE_X*ROLL_X+(tw-1))+") {\n");
    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT+"; ++i) {\n");
    		buf.append("sO[sharedIndex * "+CharacterTemplate.LINE_HEIGHT+" + i] = Ob[(blockIdx.x * "+BLOCK_SIZE_X*ROLL_X+" + sharedIndex) * "+CharacterTemplate.LINE_HEIGHT+" + i];\n");
    		buf.append("}\n");
    		buf.append("}\n");
    		buf.append("__syncthreads();\n");
    		
    		buf.append("if (Tindex < Tlength) {\n");
    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT*tw+"; ++i) {\n");
    		buf.append("float tb = Tb[Tindex * "+CharacterTemplate.LINE_HEIGHT*tw+" + i];\n");
    		for (int r=0; r<ROLL_X; ++r) {
    			buf.append("score"+r+" = __fmaf_ru(sO[(threadIdx.x * "+ROLL_X+" + "+r+") * "+CharacterTemplate.LINE_HEIGHT+" + i], tb, score"+r+");\n");
    		}
    		buf.append("}\n");
    		buf.append("int Oindex;\n");
    		for (int r=0; r<ROLL_X; ++r) {
    			buf.append("Oindex = blockIdx.x * "+BLOCK_SIZE_X*ROLL_X+" + threadIdx.x * "+ROLL_X+" + "+r+";\n");
    			buf.append("if (Oindex < Olength) scores[scoresOffset + Oindex * Tlength + Tindex] = score"+r+";\n");
    		}
    		
    		buf.append("}\n");

    		buf.append("}\n");
    	}
    	return buf.toString();
    }
	
    public String kernelSrcPrivate() {
    	StringBuffer buf = new StringBuffer();
    	for (int tw=1; tw<=CharacterTemplate.LINE_HEIGHT; ++tw) {
    		buf.append("extern \"C\"\n");
    		buf.append("__global__ void compute_emissions_"+tw+"(int scoresOffset, int Olength, int Tlength, float const* __restrict__ Tw, float const* __restrict__ Tb, float const* __restrict__ Ow, float const* __restrict__ Ob, float* scores) {\n");
    		
    		buf.append("int Tindex = blockIdx.y * "+BLOCK_SIZE_Y+" + threadIdx.y;\n");
    		buf.append("if (Tindex < Tlength) {\n");
    		
    		buf.append("float pO["+(ROLL_X+(tw-1))*CharacterTemplate.LINE_HEIGHT+"];\n");
    		buf.append("for (int r=0; r<"+(ROLL_X+(tw-1))+"; ++r) {\n");
    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT+"; ++i) {\n");
    			buf.append("pO[r * "+CharacterTemplate.LINE_HEIGHT+" + i] = Ow[(blockIdx.x * "+BLOCK_SIZE_X*ROLL_X+" + threadIdx.x * "+ROLL_X+" + r) * "+CharacterTemplate.LINE_HEIGHT+" + i];\n");
    		buf.append("}\n");
    		buf.append("}\n");
    		
    		for (int r=0; r<ROLL_X; ++r) {
    			buf.append("float score"+r+" = 0;\n");
    		}
    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT*tw+"; ++i) {\n");
    		buf.append("float tw = Tw[Tindex * "+CharacterTemplate.LINE_HEIGHT*tw+" + i];\n");
    		for (int r=0; r<ROLL_X; ++r) {
    			buf.append("score"+r+" = __fmaf_ru(pO["+(r*CharacterTemplate.LINE_HEIGHT)+" + i], tw, score"+r+");\n");
    		}
    		buf.append("}\n");

    		buf.append("for (int r=0; r<"+(ROLL_X+(tw-1))+"; ++r) {\n");
    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT+"; ++i) {\n");
    			buf.append("pO[r * "+CharacterTemplate.LINE_HEIGHT+" + i] = Ob[(blockIdx.x * "+BLOCK_SIZE_X*ROLL_X+" + threadIdx.x * "+ROLL_X+" + r) * "+CharacterTemplate.LINE_HEIGHT+" + i];\n");
    		buf.append("}\n");
    		buf.append("}\n");
    		
    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT*tw+"; ++i) {\n");
    		buf.append("float tb = Tb[Tindex * "+CharacterTemplate.LINE_HEIGHT*tw+" + i];\n");
    		for (int r=0; r<ROLL_X; ++r) {
    			buf.append("score"+r+" = __fmaf_ru(pO["+(r*CharacterTemplate.LINE_HEIGHT)+" + i], tb, score"+r+");\n");
    		}
    		buf.append("}\n");
    		buf.append("int Oindex;\n");
    		for (int r=0; r<ROLL_X; ++r) {
    			buf.append("Oindex = blockIdx.x * "+BLOCK_SIZE_X*ROLL_X+" + threadIdx.x * "+ROLL_X+" + "+r+";\n");
    			buf.append("if (Oindex < Olength) scores[scoresOffset + Oindex * Tlength + Tindex] = score"+r+";\n");
    		}
    		
    		buf.append("}\n");
    		buf.append("}\n");
    	}
    	return buf.toString();
    }

}
