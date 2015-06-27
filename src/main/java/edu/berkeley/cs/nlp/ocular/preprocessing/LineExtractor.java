package edu.berkeley.cs.nlp.ocular.preprocessing;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.Visualizer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.preprocessing.VerticalProfile.VerticalSegmentation;
import tuple.Pair;
import fileio.f;

public class LineExtractor {
	
	public static List<double[][]> extractLines(double[][] levels) {
		VerticalProfile verticalProfile = new VerticalProfile(levels);
		VerticalModel trainedModel = verticalProfile.runEM(5, 100);
		trainedModel.freezeSizeParams(1);
		VerticalSegmentation viterbiSegments = verticalProfile.decode(trainedModel);
//		ImageUtils.display(Visualizer.renderLineExtraction(levels, viterbiSegments));
		List<Pair<Integer,Integer>> lineBoundaries = viterbiSegments.retrieveLineBoundaries();
		List<double[][]> result = new ArrayList<double[][]>();
		for (Pair<Integer,Integer> boundary : lineBoundaries) {
			double[][] line = new double[levels.length][boundary.getSecond().intValue() - boundary.getFirst().intValue()];
			for (int y = boundary.getFirst().intValue(); y < boundary.getSecond().intValue(); y++) {
				for (int x = 0; x < levels.length; x++) {
					line[x][y-boundary.getFirst()] = levels[x][y];
				}
			}
			result.add(line);
		}
		System.out.println("Extractor returned " + result.size() + " line images");
		return result;
	}

	public static void main(String[] args) {
		String path = "/Users/tberg/Dropbox/corpora/ocr_data/autocrop_dev/";
		File dir = new File(path);
		for (String name : dir.list()) {
			double[][] levels = ImageUtils.getLevels(f.readImage(path+"/"+name));
//			ImageUtils.display(ImageUtils.makeImage(levels));
			double[][] rotLevels = Straightener.straighten(levels);
//			ImageUtils.display(ImageUtils.makeImage(rotLevels));
			double[][] cropLevels = Cropper.crop(rotLevels, 0.12);
			Binarizer.binarizeGlobal(0.12, cropLevels);
//			ImageUtils.display(ImageUtils.makeImage(cropLevels));
//			List<double[][]> lines = extractLines(cropLevels);
//			for (double[][] line : lines) {
//				ImageUtils.display(ImageUtils.makeImage(line));
//			}
		}
	}
	
}
