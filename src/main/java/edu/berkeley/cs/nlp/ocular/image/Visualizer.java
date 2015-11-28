package edu.berkeley.cs.nlp.ocular.image;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.preprocessing.VerticalProfile.VerticalSegmentation;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class Visualizer {

	public static BufferedImage renderObservationsAndSegmentation(PixelType[][][] observations, List<Integer>[] boundaries) {
		int[][][] rgbImage = convertToThresholdedRgb(observations, Color.BLACK.getRGB(), Color.RED.getRGB(), Color.WHITE.getRGB());
		overlaySegmentationBoundaries(rgbImage, boundaries);
		return ImageUtils.makeRgbImage(combineLinesRGB(rgbImage));
	}

	public static BufferedImage renderBlackProbs(double[][][] blackProbs) {
		return ImageUtils.makeRgbImage(combineLinesRGB(convertBlackProbsToRgb(blackProbs)));
	}

	public static BufferedImage renderObservations(PixelType[][][] observations) {
		int[][][] rgbImage = convertToThresholdedRgb(observations, Color.BLACK.getRGB(), Color.RED.getRGB(), Color.WHITE.getRGB());
		return ImageUtils.makeRgbImage(combineLinesRGB(rgbImage));
	}
	
	public static BufferedImage renderBlackProbsAndSegmentation(double[][][] blackProbs, List<Integer>[] boundaries) {
		int[][][] rgbImage = convertBlackProbsToRgb(blackProbs);
		overlaySegmentationBoundaries(rgbImage, boundaries);
		return ImageUtils.makeRgbImage(combineLinesRGB(rgbImage));
	}
	
	public static BufferedImage renderOverlay(PixelType[][][] observations, double[][][] blackProbs, List<Integer>[] boundaries) {
		int[][][] rgbImage = convertToThresholdedRgb(observations, Color.BLUE.getRGB(), Color.RED.getRGB(), Color.BLACK.getRGB());
		overlayBlackProbabilities(rgbImage, blackProbs);
		overlaySegmentationBoundaries(rgbImage, boundaries);
		return ImageUtils.makeRgbImage(combineLinesRGB(rgbImage));
	}
	
	public static BufferedImage renderLineExtraction(PixelType[][][] observations) {
		int maxWidth = 0;
		for (PixelType[][] line : observations) {
			maxWidth = Math.max(maxWidth, line.length);
		}
		int[][] rgbSpace = new int[maxWidth][observations[0][0].length];
		for (int w=0; w<maxWidth; ++w) Arrays.fill(rgbSpace[w], Color.LIGHT_GRAY.getRGB());
		
		int[][][] rgbLines = convertToThresholdedRgb(observations, Color.BLACK.getRGB(), Color.GRAY.getRGB(), Color.WHITE.getRGB());
		int[][][] rgbSpacedLines = new int[rgbLines.length*2][maxWidth][10];
		for (int line=0; line<rgbLines.length; ++line) {
			rgbSpacedLines[2*line] = rgbLines[line];
			rgbSpacedLines[2*line+1] = rgbSpace;
		}
		return ImageUtils.makeRgbImage(combineLinesRGB(rgbSpacedLines));
	}
	
	public static BufferedImage renderLineExtraction(PixelType[][] line) {
		int[][] rgbLine = convertToThresholdedRgb(new PixelType[][][]{ line }, Color.BLACK.getRGB(), Color.GRAY.getRGB(), Color.WHITE.getRGB())[0];
		return ImageUtils.makeRgbImage(rgbLine);
	}
	
	public static BufferedImage renderLineExtraction(double[][] levels, VerticalSegmentation segmentation) {
		int[][] rgbImage = new int[levels.length][levels[0].length];
		for (int i = 0; i < levels.length; i++) {
			for (int j = 0; j < levels[i].length; j++) {
				if (ImageUtils.getPixelType(levels[i][j]) == PixelType.BLACK) {
					switch (segmentation.getType(j)) {
					case ASCENDER: rgbImage[i][j] = Color.BLUE.getRGB();
					break;
					case BASE: rgbImage[i][j] = Color.BLACK.getRGB();
					break;
					case DESCENDER: rgbImage[i][j] = Color.RED.getRGB();
					break;
					default: throw new RuntimeException("Unrecognized type: " + segmentation.getType(j));
					}
				} else {
					switch (segmentation.getType(j)) {
					case ASCENDER: rgbImage[i][j] = Color.WHITE.getRGB();
					break;
					case BASE: rgbImage[i][j] = Color.WHITE.getRGB();
					break;
					case DESCENDER: rgbImage[i][j] = Color.LIGHT_GRAY.getRGB();
					break;
					default: throw new RuntimeException("Unrecognized type: " + segmentation.getType(j));
					}
				}
			}
		}
		return ImageUtils.makeRgbImage(rgbImage);
	}
	
	private static int[][][] convertToThresholdedRgb(PixelType[][][] observations, int blackColorRgb, int unobservedColorRgb, int whiteColorRgb) {
		int[][][] rgbImage = new int[observations.length][][];
		for (int i=0; i<observations.length; ++i) {
			rgbImage[i] = new int[observations[i].length][];
			for (int j=0; j<observations[i].length; ++j) {
				rgbImage[i][j] = new int[observations[i][j].length];
				for (int k=0; k<observations[i][j].length; ++k) {
					if (observations[i][j][k] == PixelType.BLACK) {
						rgbImage[i][j][k] = blackColorRgb;
					} else if (observations[i][j][k] == PixelType.WHITE) {
						rgbImage[i][j][k] = whiteColorRgb;
					} else {
						rgbImage[i][j][k] = unobservedColorRgb;
					}
				}
			}
		}
		return rgbImage;
	}

	private static int[][][] convertBlackProbsToRgb(double[][][] probs) {
		int[][][] newImage = new int[probs.length][][];
		for (int i=0; i<probs.length; ++i) {
			newImage[i] = new int[probs[i].length][];
			for (int j=0; j<probs[i].length; ++j) {
				newImage[i][j] = new int[probs[i][j].length];
				for (int k=0; k<probs[i][j].length; ++k) {
					int val = (int) (255 * (1.0 - probs[i][j][k]));
					newImage[i][j][k] = (new Color(val, val, val)).getRGB();
				}
			}
		}
		return newImage;
	}

	private static void overlaySegmentationBoundaries(int[][][] image, List<Integer>[] boundaries) {
		for (int d=0; d<image.length; ++d) {
			List<Integer> boundariesThisLine = boundaries[d];
			for (int i=0; i<boundariesThisLine.size(); ++i) {
				int boundary = boundariesThisLine.get(i);
				for (int h=0; h<image[d][0].length; ++h) {
					Color cur = new Color(image[d][boundary][h]);
					image[d][boundary][h] = (new Color((int) (Math.min(255, (cur.getRed()+50))*0.5), (int) Math.min(255, cur.getGreen()+50), (int) (Math.min(255, (cur.getBlue()+50))*0.5))).getRGB();
				}
			}
		}
	}

	private static void overlayBlackProbabilities(int[][][] image, double[][][] blackProbs) {
		for (int i=0; i<image.length; ++i) {
			for (int j=0; j<image[i].length; ++j) {
				for (int k=0; k<image[i][j].length; ++k) {
					// Swap out the redness component of the image with a redness component
					// proportional to the amount of black in the image
					int redness = (int)(blackProbs[i][j][k] * ImageUtils.MAX_LEVEL);
					image[i][j][k] = (image[i][j][k] & 0xFF00FFFF) | (redness << 16);
				}
			}
		}
	}

	private static int[][] combineLinesRGB(int[][][] imageByLine) {
		int lineHeight = imageByLine[0][0].length;
		int linesHeight = imageByLine.length * lineHeight;
		int longestLineLength = 0;
		for (int d=0; d<imageByLine.length; ++d) {
			longestLineLength = Math.max(longestLineLength, imageByLine[d].length);
		}
		int[][] combinedLines = new int[longestLineLength][linesHeight];
		for (int d=0; d<imageByLine.length; ++d) {
			for (int w=0; w<combinedLines.length; ++w) {
				int verticalOffset = d * lineHeight;
				for (int h=0; h<lineHeight; ++h) {
					// If we're still writing the line
					if (w<imageByLine[d].length) {
						combinedLines[w][h+verticalOffset] = imageByLine[d][w][h];
					} else {
						combinedLines[w][h+verticalOffset] = Color.LIGHT_GRAY.getRGB();
					}
				}
			}
		}
		return combinedLines;
	}

}
