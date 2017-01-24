package edu.berkeley.cs.nlp.ocular.preprocessing;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.image.Visualizer;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.preprocessing.VerticalProfile.VerticalSegmentation;
import tberg.murphy.tuple.Pair;
import tberg.murphy.fileio.f;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class LineExtractor {
	
	public static List<double[][]> extractLines(double[][] levels) {
		VerticalProfile verticalProfile = new VerticalProfile(levels);
		VerticalModel trainedModel = verticalProfile.runEM(5, 100);
		trainedModel.freezeSizeParams(1);
		VerticalSegmentation viterbiSegments = verticalProfile.decode(trainedModel);
		//ImageUtils.display(Visualizer.renderLineExtraction(levels, viterbiSegments));
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
		//String path = "/Users/tberg/Dropbox/corpora/ocr_data/autocrop_dev/";
		String pathAndName = "/Users/dhg/workspace/ocular/sample_images/advertencias/pl_blac_047_00039-800.jpg";
		//File dir = new File(path);
		//for (String name : dir.list()) {
			//String pathAndName = path+"/"+name;
			double[][] levels = ImageUtils.getLevels(f.readImage(pathAndName));
//			ImageUtils.display(ImageUtils.makeImage(levels));
			double[][] rotLevels = Straightener.straighten(levels);
//			ImageUtils.display(ImageUtils.makeImage(rotLevels));
			double[][] cropLevels = Cropper.crop(rotLevels, 0.12);
			Binarizer.binarizeGlobal(0.12, cropLevels);
//			ImageUtils.display(ImageUtils.makeImage(cropLevels));
			List<double[][]> lines = extractLines(cropLevels);

			PixelType[][][] observations = new PixelType[lines.size()][][];
			for (int i = 0; i < lines.size(); ++i) 
				observations[i] = ImageUtils.getPixelTypes(ImageUtils.makeImage(lines.get(i)));
			ImageUtils.display(Visualizer.renderLineExtraction(observations));
			
			//			for (double[][] line : lines) {
//				ImageUtils.display(ImageUtils.makeImage(line));
//			}
		//}
	}
	
}
