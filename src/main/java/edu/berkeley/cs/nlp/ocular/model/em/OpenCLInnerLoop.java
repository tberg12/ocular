package edu.berkeley.cs.nlp.ocular.model.em;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLPlatform.DeviceFeature;

import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import gpu.CudaUtil;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class OpenCLInnerLoop implements EmissionCacheInnerLoop {

	public static final int NVIDIA_GPU_BLOCK_SIZE_X = 1;
	public static final int NVIDIA_GPU_ROLL_X = 32;
	public static final int NVIDIA_GPU_BLOCK_SIZE_Y = 64;
	
	public static final int INTEL_GPU_BLOCK_SIZE_X = 1;
	public static final int INTEL_GPU_ROLL_X = 32;
	public static final int INTEL_GPU_BLOCK_SIZE_Y = 32;
	
	public static final int CPU_BLOCK_SIZE_X = 1;
	public static final int CPU_ROLL_X = 8;
	public static final int CPU_BLOCK_SIZE_Y = 1;
	
	private static class PointerCapturer {
		LinkedList<Pointer<?>> activePointers = new LinkedList<Pointer<?>>();
		public <T> Pointer<T> capture(Pointer<T> p) {
			activePointers.add(p);
			return p;
		}
		public void releaseAll() {
			activePointers = new LinkedList<Pointer<?>>();
		}
	}
	
	int blockSizeX;
	int rollX;
	int blockSizeY;
	int numThreads;
	int[] templateNumIndices;
	int[] templateIndicesOffsets;
	int maxTemplateWidth;
	int minTemplateWidth;
	CLContext context;
	CLQueue queue;
	CLProgram program;
	CLBuffer<Float> d_Ow;
	CLBuffer<Float> d_Ob;
	CLBuffer<Float> d_scores;
	CLBuffer<Float>[] d_Tw;
	CLBuffer<Float>[] d_Tb;
	PointerCapturer pc;

	public OpenCLInnerLoop(int numThreads) {
		// choose device
		List<CLDevice> devices = new ArrayList<CLDevice>();
		System.out.println();
	    for(CLPlatform platform : JavaCL.listPlatforms()) {
	        for (CLDevice device : platform.listAllDevices(true)) {
	        	System.out.println("Type: "+device.getType());
	        	System.out.println("Vendor: "+device.getVendor());
	        	System.out.println("Name: "+device.getName());
	        	System.out.println("Compute units: "+device.getMaxComputeUnits());
	        	System.out.println("Global mem: "+device.getGlobalMemSize()/1e6+"MB");
	        	System.out.println("Driver version: "+device.getDriverVersion());
	        	System.out.println();
	        	devices.add(device);
	        }
	    }
	    if (context == null) {
	    	for (CLDevice device : devices) {
	    		if (device.getVendor().toLowerCase().contains("intel") && device.getType().contains(CLDevice.Type.GPU) && device.getMaxComputeUnits() >= 140 && device.getGlobalMemSize() > 512e6) {
	    			this.context = JavaCL.createContext(null, device);
	    		}
	    	}
	    }
//	    if (context == null) {
//	    	for (CLDevice device : devices) {
//	    		if (device.getVendor().toLowerCase().contains("nvidia") && device.getType().contains(CLDevice.Type.GPU) && device.getMaxComputeUnits() >= 8 && device.getGlobalMemSize() > 1e9 && !device.getPlatform().getName().toLowerCase().contains("apple")) {
//	    			this.context = JavaCL.createContext(null, device);
//	    		}
//	    	}
//	    }
	    if (context == null) {
			this.context = JavaCL.createBestContext(DeviceFeature.CPU);
	    }
	    if (context.getDevices()[0].getType().contains(CLDevice.Type.GPU) && context.getDevices()[0].getVendor().toLowerCase().contains("nvidia")) {
	    	this.blockSizeX = NVIDIA_GPU_BLOCK_SIZE_X;
	    	this.rollX = NVIDIA_GPU_ROLL_X;
	    	this.blockSizeY = NVIDIA_GPU_BLOCK_SIZE_Y;
	    } else if (context.getDevices()[0].getType().contains(CLDevice.Type.GPU) && context.getDevices()[0].getVendor().toLowerCase().contains("intel")) {
	    	this.blockSizeX = INTEL_GPU_BLOCK_SIZE_X;
	    	this.rollX = INTEL_GPU_ROLL_X;
	    	this.blockSizeY = INTEL_GPU_BLOCK_SIZE_Y;
	    } else if (context.getDevices()[0].getType().contains(CLDevice.Type.CPU)) {
	    	this.blockSizeX = CPU_BLOCK_SIZE_X;
	    	this.rollX = CPU_ROLL_X;
	    	this.blockSizeY = CPU_BLOCK_SIZE_Y;
	    }
	    System.out.println("Using context:");
	    System.out.println(context.toString());
	    System.out.println("Block size x: "+blockSizeX);
	    System.out.println("Roll x: "+rollX);
	    System.out.println("Block size y: "+blockSizeY);
		

        this.context.setCacheBinaries(false);
        this.queue = context.createDefaultQueue();
        this.program = context.createProgram(kernelSrc());
        this.program.addBuildOption("-cl-fast-relaxed-math");
        this.program.addBuildOption("-cl-mad-enable");
        this.program.addBuildOption("-cl-unsafe-math-optimizations");
        this.program.addBuildOption("-cl-fast-relaxed-math");
        this.program.addBuildOption("-cl-single-precision-constant");
        this.program.build();
        
		this.pc = new PointerCapturer();
		this.numThreads = numThreads;
	}
	
	@SuppressWarnings("unchecked")
	public void startup(float[][] whiteTemplates, float[][] blackTemplates, int[] templateNumIndices, int[] templateIndicesOffsets, int minTemplateWidth, int maxTemplateWidth, int maxSequenceLength, int totalTemplateNumIndices) {
		this.templateNumIndices = templateNumIndices;
		this.templateIndicesOffsets = templateIndicesOffsets;
		this.maxTemplateWidth = maxTemplateWidth;
		this.minTemplateWidth = minTemplateWidth;
		
		// Allocate the device input data
		int extendedMaxSeqLength = (blockSizeX*rollX) * (int) Math.ceil(((double) maxSequenceLength) / (blockSizeX*rollX));
		this.d_Ow = context.createFloatBuffer(Usage.Input, (extendedMaxSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT);
		this.d_Ob = context.createFloatBuffer(Usage.Input, (extendedMaxSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT);
		this.d_scores = context.createFloatBuffer(Usage.Output, maxSequenceLength*totalTemplateNumIndices);
		
		int numTemplateWidths = (maxTemplateWidth-minTemplateWidth)+1;
		this.d_Tw = new CLBuffer[numTemplateWidths];
		this.d_Tb = new CLBuffer[numTemplateWidths];
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				d_Tw[tw-minTemplateWidth] = context.createFloatBuffer(Usage.Input, whiteTemplates[tw-minTemplateWidth].length);
				d_Tw[tw-minTemplateWidth].write(queue, pc.capture(Pointer.pointerToFloats(whiteTemplates[tw-minTemplateWidth])), false);
				
				d_Tb[tw-minTemplateWidth] = context.createFloatBuffer(Usage.Input, whiteTemplates[tw-minTemplateWidth].length);
				d_Tb[tw-minTemplateWidth].write(queue, pc.capture(Pointer.pointerToFloats(blackTemplates[tw-minTemplateWidth])), false);
			}
		}
	}

	public void shutdown() {
		d_Ow.release();
		d_Ob.release();
		d_scores.release();
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				d_Tw[tw-minTemplateWidth].release();
				d_Tb[tw-minTemplateWidth].release();
			}
		}
		pc.releaseAll();
	}

	public void compute(final float[] scores, final float[] whiteObservations, final float[] blackObservations, final int sequenceLength) {
		int gridSizeX = (int) Math.ceil(((double) sequenceLength) / (blockSizeX*rollX));
		int extendedSeqLength = gridSizeX * (blockSizeX*rollX);
		d_Ow.write(queue, pc.capture(Pointer.pointerToFloats(CudaUtil.extendWithZeros(whiteObservations, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT))), false);
		d_Ob.write(queue, pc.capture(Pointer.pointerToFloats(CudaUtil.extendWithZeros(blackObservations, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT))), false);
		queue.enqueueBarrier();
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				int gridSizeY = (int) Math.ceil(((double) templateNumIndices[tw-minTemplateWidth]) / blockSizeY);
		        CLKernel computeKernel = program.createKernel("compute_emissions_"+tw);
		        computeKernel.setArgs(templateIndicesOffsets[tw-minTemplateWidth]*sequenceLength, sequenceLength, templateNumIndices[tw-minTemplateWidth], d_Tw[tw-minTemplateWidth], d_Tb[tw-minTemplateWidth], d_Ow, d_Ob, d_scores);
		        computeKernel.enqueueNDRange(queue, new int[] {gridSizeX*blockSizeX, gridSizeY*blockSizeY}, new int[] {blockSizeX, blockSizeY});
			}
		}
		queue.enqueueBarrier();
		d_scores.read(queue).getFloats(scores);
	}

	public int numOuterThreads() {
		return 1;
	}

	public int numPopulateThreads() {
		return numThreads;
	}
	
    public String kernelSrc() {
    	StringBuffer buf = new StringBuffer();
    	for (int tw=1; tw<=CharacterTemplate.LINE_HEIGHT; ++tw) {
    		buf.append("__kernel void compute_emissions_"+tw+"(int scoresOffset, int Olength, int Tlength, __global float const* __restrict__ Tw, __global float const* __restrict__ Tb, __global float const* __restrict__ Ow, __global float const* __restrict__ Ob, __global float* scores) {\n");
    		
    		buf.append("int Tindex = get_global_id(1);\n");
    		buf.append("if (Tindex < Tlength) {\n");
    		
    		for (int r=0; r<rollX; ++r) {
    			buf.append("float o"+r+" = 0;\n");
    			buf.append("float score"+r+" = 0;\n");
    		}
    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT*tw+"; ++i) {\n");
    		buf.append("float tw = Tw[Tindex * "+CharacterTemplate.LINE_HEIGHT*tw+" + i];\n");
    		for (int r=0; r<rollX; ++r) {
    			buf.append("o"+r+" = Ow[(get_group_id(0) * "+blockSizeX*rollX+" + get_local_id(0) * "+rollX+" + "+r+") * "+CharacterTemplate.LINE_HEIGHT+" + i];\n");
//    			buf.append("score"+r+" = fma(o"+r+", tw, score"+r+");\n");
//    			buf.append("score"+r+" = mad(o"+r+", tw, score"+r+");\n");
    			buf.append("score"+r+" += o"+r+" * tw;\n");
    		}
    		buf.append("}\n");

    		buf.append("for (int i=0; i<"+CharacterTemplate.LINE_HEIGHT*tw+"; ++i) {\n");
    		buf.append("float tb = Tb[Tindex * "+CharacterTemplate.LINE_HEIGHT*tw+" + i];\n");
    		for (int r=0; r<rollX; ++r) {
    			buf.append("o"+r+" = Ob[(get_group_id(0) * "+blockSizeX*rollX+" + get_local_id(0) * "+rollX+" + "+r+") * "+CharacterTemplate.LINE_HEIGHT+" + i];\n");
//    			buf.append("score"+r+" = fma(o"+r+", tb, score"+r+");\n");
//    			buf.append("score"+r+" = mad(o"+r+", tb, score"+r+");\n");
    			buf.append("score"+r+" += o"+r+" * tb;\n");
    		}
    		buf.append("}\n");
    		buf.append("int Oindex;\n");
    		for (int r=0; r<rollX; ++r) {
    			buf.append("Oindex = get_group_id(0) * "+blockSizeX*rollX+" + get_local_id(0) * "+rollX+" + "+r+";\n");
    			buf.append("if (Oindex < Olength) scores[scoresOffset + Oindex * Tlength + Tindex] = score"+r+";\n");
    		}
    		
    		buf.append("}\n");
    		buf.append("}\n");
    	}
    	return buf.toString();
    }

}
