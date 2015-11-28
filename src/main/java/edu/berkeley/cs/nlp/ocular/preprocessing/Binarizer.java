package edu.berkeley.cs.nlp.ocular.preprocessing;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class Binarizer {

	public static boolean isBinary(double[][] levels) {
		int[] histogram = new int[(int) ImageUtils.MAX_LEVEL+1];
		for (int i=0; i<levels.length; i++) {
			for (int j=0; j<levels[i].length; j++) {
				histogram[(int) levels[i][j]]++;
			}
		}
		int nonZeroEntries = 0;
		for (int count : histogram) {
			if (count > 0) nonZeroEntries++;
		}
		return nonZeroEntries <= 2;
	}
	
	public static void binarizeAlreadyBinary(double[][] levels) {
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for (double[] vals : levels) {
			for (double val : vals) {
				min = Math.min(val, min);
				max = Math.max(val, max);
			}
		}
		double threshold = (max + min) / 2.0;
		for (int i = 0; i < levels.length; i++) {
			for (int j = 0; j < levels[i].length; j++) {
				if (levels[i][j] <= threshold) {
					levels[i][j] = 0;
				} else {
					levels[i][j] = ImageUtils.MAX_LEVEL;
				}
			}
		}
	}
	
	public static void binarizeGlobal(double blackPercential, double[][] levels) {
		if (isBinary(levels)) {
			binarizeAlreadyBinary(levels);
			return;
		}
		
		int[] histogram = new int[(int) ImageUtils.MAX_LEVEL+1];
		int total = 0;
		for (int i=0; i<levels.length; i++) {
			for (int j=0; j<levels[i].length; j++) {
				histogram[(int) levels[i][j]]++;
				total++;
			}
		}
		
		int rank = (int) (Math.ceil(total * blackPercential));
		int curRank = 0;
		double threshold = 0.0;
		for (int v=0; v<histogram.length; ++v) {
			curRank += histogram[v];
			if (curRank >= rank) {
				threshold = v;
				break;
			}
		}
		for (int i = 0; i < levels.length; i++) {
			for (int j = 0; j < levels[i].length; j++) {
				if (levels[i][j] <= threshold) {
					levels[i][j] = 0;
				} else {
					levels[i][j] = ImageUtils.MAX_LEVEL;
				}
			}
		}
	}
	
	public static void binarizeLocal(double blackPercential, double radiusFactor,  double[][] levels) {
		if (isBinary(levels)) {
			binarizeAlreadyBinary(levels);
			return;
		}
		
		int radius = (int) (levels.length * radiusFactor);
		
		int dWidth = (int) Math.ceil((double) levels.length / radius);
		int dHeight = (int) Math.ceil((double) levels[0].length / radius);
		double[][] thresholds = new double[dWidth][dHeight];
		for (int di=0; di<dWidth; ++di) {
			for (int dj=0; dj<dHeight; ++dj) {
				int i = di*radius + radius / 2;
				int j = dj*radius + radius / 2;
				if (i < levels.length && j < levels[0].length) {
					int[] histogram = new int[(int) ImageUtils.MAX_LEVEL+1];
					int total = 0;
					for (int i0 = Math.max(0,i-radius); i0 < Math.min(levels.length,i+radius); i0++) {
						for (int j0 = Math.max(0,j-radius); j0 < Math.min(levels[i].length,j+radius); j0++) {
							histogram[(int) levels[i0][j0]]++;
							total++;
						}
					}
					int rank = (int) (Math.ceil(total * blackPercential));
					int curRank = 0;
					double threshold = 0.0;
					for (int v=0; v<histogram.length; ++v) {
						curRank += histogram[v];
						if (curRank >= rank) {
							threshold = v;
							break;
						}
					}
					thresholds[di][dj] = threshold;
				}
			}
		}
		
		
		for (int i = 0; i < levels.length; i++) {
			for (int j = 0; j < levels[i].length; j++) {
				if (levels[i][j] <= thresholds[i/radius][j/radius]) {
					levels[i][j] = 0;
				} else {
					levels[i][j] = ImageUtils.MAX_LEVEL;
				}
			}
		}
	}
	
}
