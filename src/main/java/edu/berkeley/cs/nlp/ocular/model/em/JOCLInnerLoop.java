package edu.berkeley.cs.nlp.ocular.model.em;

import static org.jocl.CL.*;

import org.jocl.*;

import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import tberg.murphy.gpu.CudaUtil;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class JOCLInnerLoop implements EmissionCacheInnerLoop {

	public static final int GPU_BLOCK_SIZE_X = 1;
	public static final int GPU_ROLL_X = 32;
	public static final int GPU_BLOCK_SIZE_Y = 64;
	
	public static final int CPU_BLOCK_SIZE_X = 1;
	public static final int CPU_ROLL_X = 8;
	public static final int CPU_BLOCK_SIZE_Y = 1;
	
	int blockSizeX;
	int rollX;
	int blockSizeY;
	int numThreads;
	int[] templateNumIndices;
	int[] templateIndicesOffsets;
	int maxTemplateWidth;
	int minTemplateWidth;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_mem d_Ow;
	cl_mem d_Ob;
	cl_mem d_scores;
	cl_mem[] d_Tw;
	cl_mem[] d_Tb;
	cl_kernel[] kernels;
	
    private static String getString(cl_device_id device, int paramName) {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

	
	public JOCLInnerLoop(int numThreads) {
		this.numThreads = numThreads;
		
        final int platformIndex = 0;
        CL.setExceptionsEnabled(true);
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        
        cl_device_id device = null;
        
        boolean isGPU = false;
        {
        	int numDevicesArray[] = new int[1];
        	final long deviceType = CL_DEVICE_TYPE_GPU;
        	clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        	int numDevices = numDevicesArray[0];
        	cl_device_id devices[] = new cl_device_id[numDevices];
        	clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        	for (int i=0; i<devices.length; ++i) {
        		String deviceName = getString(devices[i], CL_DEVICE_NAME).toLowerCase();
        		if (deviceName.contains("radeon") || deviceName.contains("nvidia")) {
        			device = devices[i];
        			isGPU = true;
        			break;
        		}
        	}
        }
        
        if (!isGPU) {
        	int numDevicesArray[] = new int[1];
        	final long deviceType = CL_DEVICE_TYPE_CPU;
        	clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        	int numDevices = numDevicesArray[0];
        	cl_device_id devices[] = new cl_device_id[numDevices];
        	clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        	device = devices[0];
        	isGPU = false;
        }
        
		if (isGPU) {
			this.blockSizeX = GPU_BLOCK_SIZE_X;
			this.rollX = GPU_ROLL_X;
			this.blockSizeY = GPU_BLOCK_SIZE_Y;
		} else {
			this.blockSizeX = CPU_BLOCK_SIZE_X;
			this.rollX = CPU_ROLL_X;
			this.blockSizeY = CPU_BLOCK_SIZE_Y;
		}
		
		System.out.printf("Device name: %s\n", getString(device, CL_DEVICE_NAME));
	    System.out.println("Block size x: "+blockSizeX);
	    System.out.println("Roll x: "+rollX);
	    System.out.println("Block size y: "+blockSizeY);
        
        
        // Create a context for the selected device
        context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
        
        // Create a command-queue
        queue =  clCreateCommandQueue(context, device, 0, null);

        // Create the program from the source code
        program = clCreateProgramWithSource(context, 1, new String[]{ kernelSrc() }, null, null);
        
        // Build the program
        clBuildProgram(program, 0, null, "-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -cl-no-signed-zeros", null, null);
        
	}
	
	public void startup(float[][] whiteTemplates, float[][] blackTemplates, int[] templateNumIndices, int[] templateIndicesOffsets, int minTemplateWidth, int maxTemplateWidth, int maxSequenceLength, int totalTemplateNumIndices) {
		this.templateNumIndices = templateNumIndices;
		this.templateIndicesOffsets = templateIndicesOffsets;
		this.maxTemplateWidth = maxTemplateWidth;
		this.minTemplateWidth = minTemplateWidth;
		
		int numTemplateWidths = (maxTemplateWidth-minTemplateWidth)+1;

		// Build kernels
        kernels = new cl_kernel[numTemplateWidths];
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				kernels[tw-minTemplateWidth] = clCreateKernel(program, "compute_emissions_"+tw, null);
			}
		}
		
		// Allocate the device input data
		int extendedMaxSeqLength = (blockSizeX*rollX) * (int) Math.ceil(((double) maxSequenceLength) / (blockSizeX*rollX));
		this.d_Ow = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * (extendedMaxSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT, null, null);
//		this.d_Ow = context.createFloatBuffer(Usage.Input, (extendedMaxSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT);
		this.d_Ob = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * (extendedMaxSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT, null, null);
//		this.d_Ob = context.createFloatBuffer(Usage.Input, (extendedMaxSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT);
		this.d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * maxSequenceLength*totalTemplateNumIndices, null, null);
//		this.d_scores = context.createFloatBuffer(Usage.Output, maxSequenceLength*totalTemplateNumIndices);
		
		this.d_Tw = new cl_mem[numTemplateWidths];
		this.d_Tb = new cl_mem[numTemplateWidths];
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				d_Tw[tw-minTemplateWidth] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * whiteTemplates[tw-minTemplateWidth].length, Pointer.to(whiteTemplates[tw-minTemplateWidth]), null);
//				d_Tw[tw-minTemplateWidth] = context.createFloatBuffer(Usage.Input, whiteTemplates[tw-minTemplateWidth].length);
//				d_Tw[tw-minTemplateWidth].write(queue, pc.capture(Pointer.pointerToFloats(whiteTemplates[tw-minTemplateWidth])), false);
				
				d_Tb[tw-minTemplateWidth] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * blackTemplates[tw-minTemplateWidth].length, Pointer.to(blackTemplates[tw-minTemplateWidth]), null);
//				d_Tb[tw-minTemplateWidth] = context.createFloatBuffer(Usage.Input, whiteTemplates[tw-minTemplateWidth].length);
//				d_Tb[tw-minTemplateWidth].write(queue, pc.capture(Pointer.pointerToFloats(blackTemplates[tw-minTemplateWidth])), false);
			}
		}
	}

	public void shutdown() {
		clReleaseMemObject(d_Ow);
		clReleaseMemObject(d_Ob);
		clReleaseMemObject(d_scores);
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				clReleaseMemObject(d_Tw[tw-minTemplateWidth]);
				clReleaseMemObject(d_Tb[tw-minTemplateWidth]);
			}
		}
		for (cl_kernel kernel : kernels) clReleaseKernel(kernel);
	}

	public void compute(final float[] scores, final float[] whiteObservations, final float[] blackObservations, final int sequenceLength) {
		int numTemplateWidths = (maxTemplateWidth-minTemplateWidth)+1;
		int gridSizeX = (int) Math.ceil(((double) sequenceLength) / (blockSizeX*rollX));
		int extendedSeqLength = gridSizeX * (blockSizeX*rollX);
		cl_event[] writeEvents = new cl_event[] {new cl_event(), new cl_event()};
		clEnqueueWriteBuffer(queue, d_Ow, CL_TRUE, 0, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT * Sizeof.cl_float, Pointer.to(CudaUtil.extendWithZeros(whiteObservations, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT)), 0, null, writeEvents[0]);
//		d_Ow.write(queue, pc.capture(Pointer.pointerToFloats(CudaUtil.extendWithZeros(whiteObservations, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT))), false);
		clEnqueueWriteBuffer(queue, d_Ob, CL_TRUE, 0, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT * Sizeof.cl_float, Pointer.to(CudaUtil.extendWithZeros(blackObservations, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT)), 0, null, writeEvents[1]);
//		d_Ob.write(queue, pc.capture(Pointer.pointerToFloats(CudaUtil.extendWithZeros(blackObservations, (extendedSeqLength+maxTemplateWidth-1)*CharacterTemplate.LINE_HEIGHT))), false);
		cl_event[] kernelEvents = new cl_event[numTemplateWidths];
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			if (templateNumIndices[tw-minTemplateWidth] > 0) {
				int gridSizeY = (int) Math.ceil(((double) templateNumIndices[tw-minTemplateWidth]) / blockSizeY);
				cl_kernel kernel = kernels[tw-minTemplateWidth];
				clSetKernelArg(kernel, 0, Sizeof.cl_int, Pointer.to(new int[] {templateIndicesOffsets[tw-minTemplateWidth]*sequenceLength}));
				clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[] {sequenceLength}));
				clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] {templateNumIndices[tw-minTemplateWidth]}));
				clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(d_Tw[tw-minTemplateWidth]));
				clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(d_Tb[tw-minTemplateWidth]));
				clSetKernelArg(kernel, 5, Sizeof.cl_mem, Pointer.to(d_Ow));
				clSetKernelArg(kernel, 6, Sizeof.cl_mem, Pointer.to(d_Ob));
				clSetKernelArg(kernel, 7, Sizeof.cl_mem, Pointer.to(d_scores));
//		        computeKernel.setArgs(templateIndicesOffsets[tw-minTemplateWidth]*sequenceLength, sequenceLength, templateNumIndices[tw-minTemplateWidth], d_Tw[tw-minTemplateWidth], d_Tb[tw-minTemplateWidth], d_Ow, d_Ob, d_scores);
				kernelEvents[tw-minTemplateWidth] = new cl_event();
				clEnqueueNDRangeKernel(queue, kernel, 2, null, new long[] {gridSizeX*blockSizeX, gridSizeY*blockSizeY}, new long[] {blockSizeX, blockSizeY}, 2, writeEvents, kernelEvents[tw-minTemplateWidth]);
//		        computeKernel.enqueueNDRange(queue, new int[] {gridSizeX*blockSizeX, gridSizeY*blockSizeY}, new int[] {blockSizeX, blockSizeY});
			}
		}
		cl_event readEvent = new cl_event();
        clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, scores.length * Sizeof.cl_float, Pointer.to(scores), kernelEvents.length, kernelEvents, readEvent);
        clWaitForEvents(1, new cl_event[] {readEvent});
//		d_scores.read(queue).getFloats(scores);
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
    		buf.append("__kernel void compute_emissions_"+tw+"(__const int scoresOffset, __const int Olength, __const int Tlength, __global float const* __restrict__ Tw, __global float const* __restrict__ Tb, __global float const* __restrict__ Ow, __global float const* __restrict__ Ob, __global float* scores) {\n");
    		
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
