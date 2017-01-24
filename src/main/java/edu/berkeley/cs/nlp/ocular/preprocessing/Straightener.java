package edu.berkeley.cs.nlp.ocular.preprocessing;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.ConnectedComponentProcessor;
import tberg.murphy.fileio.f;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
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
	
	public static void main(String[] args) {
		String path = args[0];
		double binarizeThresh = 0.1;
		if (args.length > 1) {
			binarizeThresh = Double.parseDouble(args[1]);
		}
		File dir = new File(path);
		String[] names = dir.list(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				return name.endsWith(".png") || name.endsWith(".jpg");
			}
		});
		Arrays.sort(names);
		File straightDir = new File(path + "/straight");
		straightDir.mkdirs();
		for (String name : names) {
			double[][] levels = ImageUtils.getLevels(f.readImage(path+"/"+name));
			ConnectedComponentProcessor ccprocBig = new ConnectedComponentProcessor() {
				public void process(double[][] levels, List<int[]> connectedComponent) {
					if (connectedComponent.size() > 1000) {
						for (int[] pixel : connectedComponent) {
							levels[pixel[0]][pixel[1]] = 255.0;
						}
					}
				}
			};
			ImageUtils.processConnectedComponents(levels, 50.0, ccprocBig);
			Binarizer.binarizeGlobal(binarizeThresh, levels);
			ConnectedComponentProcessor ccprocSmall = new ConnectedComponentProcessor() {
				public void process(double[][] levels, List<int[]> connectedComponent) {
					if (connectedComponent.size() < 20 || connectedComponent.size() > 500) {
						for (int[] pixel : connectedComponent) {
							levels[pixel[0]][pixel[1]] = 255.0;
						}
					}
				}
			};
			ImageUtils.processConnectedComponents(levels, 127.0, ccprocSmall);
			double[][] rotLevels = Straightener.straighten(levels);
			String baseName = (name.lastIndexOf('.') == -1) ? name : name.substring(0, name.lastIndexOf('.'));
			f.writeImage(straightDir.getAbsolutePath() +"/"+ baseName + ".png", ImageUtils.makeImage(rotLevels));
		}
	}
}
