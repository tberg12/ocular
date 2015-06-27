package edu.berkeley.cs.nlp.ocular.preprocessing;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;

import java.util.Arrays;

import arrays.a;
import tuple.Pair;

public class Cropper {
	
	public static final int NUM_CROP_POINTS = 200;
	public static final double HORIZ_MIN_CENTER_SEG_RATIO = 0.6;
	public static final double VERT_MIN_CENTER_SEG_RATIO = 0.6;
	public static final double HORIZ_GROW_RATIO = 0.03;
	public static final double VERT_GROW_RATIO = 0.0;
	public static final double INIT_SEG_WEIGHT = 1.0;
	public static final double CENTER_SEG_WEIGHT = 4.0;
	public static final double FINAL_SEG_WEIGHT = 1.0;
	public static final double CONVOLVE_DIST_RATION = 0.0015;
	
	public static double[][] crop(double[][] levels, double binarizeThreshold) {
		double[][] binaryLevels = a.copy(levels);
		Binarizer.binarizeGlobal(binarizeThreshold, binaryLevels);
		
		levels = a.transpose(levels);
		binaryLevels = a.transpose(binaryLevels);
		{
			double[] varProfile = totalVariationProfile(a.transpose(convolveRows(a.transpose(binaryLevels), (int) (CONVOLVE_DIST_RATION*binaryLevels.length))));
			Pair<Integer,Integer> seg = singleColumnSegment(varProfile, HORIZ_MIN_CENTER_SEG_RATIO);
			levels = Arrays.copyOfRange(levels, Math.max(0, seg.getFirst()-(int) (levels.length*HORIZ_GROW_RATIO)), Math.min(levels.length, seg.getSecond()+(int) (levels.length*HORIZ_GROW_RATIO)));
		}
		levels = a.transpose(levels);
		binaryLevels = a.transpose(binaryLevels);
		{
			double[] varProfile = totalVariationProfile(a.transpose(convolveRows(a.transpose(binaryLevels), (int) (CONVOLVE_DIST_RATION*levels.length))));
			Pair<Integer,Integer> seg = singleColumnSegment(varProfile, VERT_MIN_CENTER_SEG_RATIO);
			levels = Arrays.copyOfRange(levels, Math.max(0, seg.getFirst()-(int) (levels.length*HORIZ_GROW_RATIO)), Math.min(levels.length, seg.getSecond()+(int) (levels.length*HORIZ_GROW_RATIO)));
		}
		return levels;
	}
	
	private static double[] totalVariationProfile(double[][] levels) {
		double[] varProfile = new double[levels.length];
		for (int i=0; i<levels.length; ++i) {
			for (int j=0; j<levels[i].length-1; ++j) {
				varProfile[i] += Math.abs(levels[i][j] - levels[i][j+1]) / (levels[i].length-1);
			}
		}
		return varProfile;
	}
	
	public static double[][] convolveRows(double[][] binarizedLevels, int pixels) {
		double[][] result = new double[binarizedLevels.length][binarizedLevels[0].length];
		for (int i=0; i<result.length; ++i) {
			Arrays.fill(result[i], ImageUtils.MAX_LEVEL);
		}
		for (int i=0; i<binarizedLevels.length; ++i) {
			for (int j=0; j<binarizedLevels[i].length; ++j) {
				if (binarizedLevels[i][j] < ImageUtils.MAX_LEVEL) {
					for (int k=Math.max(0,j-pixels); k<=Math.min(binarizedLevels[i].length-1,j+pixels); ++k) {
						result[i][k] = 0;
					}
				}
			}
		}
		return result;
	}
	
	private static Pair<Integer,Integer> singleColumnSegment(double[] varProfile, double minCenterWidthFrac) {
		int minCenterWidth = (int) (minCenterWidthFrac * varProfile.length);
		int bestI = -1;
		int bestJ = -1;
		double bestObjective = Double.POSITIVE_INFINITY;
		for (int i=0; i<varProfile.length; i+=varProfile.length/NUM_CROP_POINTS) {
			for (int j=i+minCenterWidth; j<varProfile.length; j+=varProfile.length/NUM_CROP_POINTS) {
				double val = evalSingleSegmentation(i, j, varProfile);
				if (val < bestObjective) {
					bestObjective = val;
					bestI = i;
					bestJ = j;
				}
			}
		}
		return Pair.makePair(bestI, bestJ);
	}
	
	private static double evalSingleSegmentation(int i, int j, double[] totalVars) {
		double result = 0.0;
		{
			double mean = 0.0;
			for (int ii=0; ii<i; ++ii) {
				mean += totalVars[ii] / i;
			}
			double var = 0.0;
			for (int ii=0; ii<i; ++ii) {
				double diff = (mean - totalVars[ii]);
				var += diff*diff / i;
			}
			result += INIT_SEG_WEIGHT * var;
		}
		{
			double mean = 0.0;
			for (int ii=i; ii<j; ++ii) {
				mean += totalVars[ii] / (j-i);
			}
			double var = 0.0;
			for (int ii=i; ii<j; ++ii) {
				double diff = (mean - totalVars[ii]);
				var += diff*diff / (j-i);
			}
			result += CENTER_SEG_WEIGHT * var;
		}
		{
			double mean = 0.0;
			for (int ii=j; ii<totalVars.length; ++ii) {
				mean += totalVars[ii] / (totalVars.length - j);
			}
			double var = 0.0;
			for (int ii=j; ii<totalVars.length; ++ii) {
				double diff = (mean - totalVars[ii]);
				var += diff*diff / (totalVars.length - j);
			}
			result += FINAL_SEG_WEIGHT * var;
		}
		return result;
	}
	
//	private static int[] doubleColumnSegment(double[] varProfile, double minCenterWidthFrac) {
//		int minCenterWidth = (int) (minCenterWidthFrac * varProfile.length);
//		int bestI = -1;
//		int bestJ = -1;
//		int bestC = -1;
//		double bestObjective = Double.POSITIVE_INFINITY;
//		for (int i=0; i<varProfile.length; i+=varProfile.length/NUM_CROP_POINTS) {
//			for (int j=i+minCenterWidth; j<varProfile.length; j+=varProfile.length/NUM_CROP_POINTS) {
//				for (int c=i+1; c<j; c+=varProfile.length/NUM_CROP_POINTS++) {
//					double val = evalDoubleSegmentation(i, j, c, varProfile);
//					if (val < bestObjective) {
//						bestObjective = val;
//						bestI = i;
//						bestJ = j;
//					}
//				}
//			}
//		}
//		return new int[] {bestI, bestJ, bestC};
//	}
	
//	private static double evalDoubleSegmentation(int i, int j, int c, double[] totalVars) {
//		double result = 0.0;
//		{
//			double mean = 0.0;
//			for (int ii=0; ii<i; ++ii) {
//				mean += totalVars[ii] / i;
//			}
//			double var = 0.0;
//			for (int ii=0; ii<i; ++ii) {
//				double diff = (mean - totalVars[ii]);
//				var += diff*diff / i;
//			}
//			result += INIT_SEG_WEIGHT * var;
//		}
//		{
//			double mean = 0.0;
//			for (int ii=i; ii<j; ++ii) {
//				mean += totalVars[ii] / (j-i);
//			}
//			double var = 0.0;
//			for (int ii=i; ii<j; ++ii) {
//				double diff = (mean - totalVars[ii]);
//				var += diff*diff / (j-i);
//			}
//			result += CENTER_SEG_WEIGHT * var;
//		}
//		{
//			double mean = 0.0;
//			for (int ii=j; ii<totalVars.length; ++ii) {
//				mean += totalVars[ii] / (totalVars.length - j);
//			}
//			double var = 0.0;
//			for (int ii=j; ii<totalVars.length; ++ii) {
//				double diff = (mean - totalVars[ii]);
//				var += diff*diff / (totalVars.length - j);
//			}
//			result += FINAL_SEG_WEIGHT * var;
//		}
//		return result;
//	}
	
}
