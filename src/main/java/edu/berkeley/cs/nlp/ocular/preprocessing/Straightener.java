package edu.berkeley.cs.nlp.ocular.preprocessing;

import java.awt.image.BufferedImage;
import java.io.File;

import fileio.f;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils;

public class Straightener {
	
	private static final double MIN_ANGLE_RADIANS = -0.05;
	private static final double MAX_ANGLE_RADIANS = 0.05;
	private static final int ANGLE_SAMPLE_POINTS = 20;
	
	public static double[][] straighten(double[][] levels) {
		BufferedImage image = ImageUtils.makeImage(levels);
		double maxTotalVar = Double.NEGATIVE_INFINITY;
		double bestAngle = Double.NEGATIVE_INFINITY;
		for (int i=0; i<ANGLE_SAMPLE_POINTS; ++i) {
			double angle = MIN_ANGLE_RADIANS + ((double) i / (ANGLE_SAMPLE_POINTS-1)) * (MAX_ANGLE_RADIANS - MIN_ANGLE_RADIANS);
			double[][] rotLevels = ImageUtils.getLevels(ImageUtils.rotateImage(image, angle));
			double totalVar = verticalTotalVariation(rotLevels);
//			System.out.println("angle: "+angle+", total var: "+totalVar);
			if (totalVar > maxTotalVar) {
				maxTotalVar = totalVar;
				bestAngle = angle;
			}
		}
		return ImageUtils.getLevels(ImageUtils.rotateImage(ImageUtils.makeImage(levels), bestAngle));
	}
	
	private static double verticalTotalVariation(double[][] levels) {
		double[] horizontalAvg = new double[levels[0].length];
		for (int i=0; i<levels.length; ++i) {
			for (int j=0; j<levels[0].length; ++j) {
				horizontalAvg[j] += levels[i][j] / levels.length;
			}
		}
		double totalVar = 0;
		for (int j=1; j<levels[0].length; ++j) {
			totalVar += Math.abs(horizontalAvg[j] - horizontalAvg[j-1]);
		}
		return totalVar / (levels[0].length-1);
	}
	
}
