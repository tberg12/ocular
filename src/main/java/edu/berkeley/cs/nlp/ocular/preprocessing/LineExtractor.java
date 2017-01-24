package edu.berkeley.cs.nlp.ocular.preprocessing;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.ConnectedComponentProcessor;
import edu.berkeley.cs.nlp.ocular.preprocessing.VerticalProfile.VerticalSegmentation;
import tberg.murphy.fileio.f;
import tberg.murphy.tuple.Pair;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class LineExtractor {
	
	public static List<double[][]> extractLines(double[][] levels) {
		VerticalProfile verticalProfile = new VerticalProfile(levels);
		VerticalModel trainedModel = verticalProfile.runEM(5, 100);
//		trainedModel.freezeSizeParams(1);
		VerticalSegmentation viterbiSegments = verticalProfile.decode(trainedModel);
//		ImageUtils.display(Visualizer.renderLineExtraction(levels, viterbiSegments));
		
		List<double[][]> result = new ArrayList<double[][]>();

		int topDist = 29;
		int botDist = 11;
		List<Pair<Integer,Integer>> segments = viterbiSegments.retrieveLineBoundaries();
		List<Integer> baselines = viterbiSegments.retrieveBaselines();
		for (int s=0; s<baselines.size(); ++s) {
			int base = baselines.get(s);
			int upper = segments.get(s).getFirst();
			int lower = segments.get(s).getSecond();
			double[][] line = new double[levels.length][topDist+botDist];
			for (int t=0; t<topDist; ++t) {
				for (int x = 0; x < levels.length; x++) {
					int pos = base+(t-topDist);
					if (pos < 0 || pos >= levels[0].length){
//						if (pos < 0 || pos >= levels[0].length || pos < upper-5 || pos >= lower+5){
						line[x][t] = ImageUtils.MAX_LEVEL;
					} else {
						line[x][t] = levels[x][pos];
					}
				}
			}
			for (int b=0; b<botDist; ++b) {
				for (int x = 0; x < levels.length; x++) {
					int pos = base+b;
					if (pos < 0 || pos >= levels[0].length){
//						if (pos < 0 || pos >= levels[0].length || pos < upper-5 || pos >= lower+5){
						line[x][topDist+b] = ImageUtils.MAX_LEVEL;
					} else {
						line[x][topDist+b] = levels[x][pos];
					}
				}
			}
			result.add(line);
		}
		
//		List<Pair<Integer,Integer>> lineBoundaries = viterbiSegments.retrieveLineBoundaries();
//		for (Pair<Integer,Integer> boundary : lineBoundaries) {
//			double[][] line = new double[levels.length][boundary.getSecond().intValue() - boundary.getFirst().intValue()];
//			for (int y = boundary.getFirst().intValue(); y < boundary.getSecond().intValue(); y++) {
//				for (int x = 0; x < levels.length; x++) {
//					line[x][y-boundary.getFirst()] = levels[x][y];
//				}
//			}
//			result.add(line);
//		}
		
		System.out.println("Extractor returned " + result.size() + " line images");
		return result;
	}

	public static void main(String[] args) {
		String path = "/Users/tberg/Desktop/F-tem/seg_extraction/";
		File dir = new File(path);
		for (String name : dir.list(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				return name.endsWith(".png") || name.endsWith(".jpg");
			}
		})) {
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
			Binarizer.binarizeGlobal(0.13, levels);
			ConnectedComponentProcessor ccprocSmall = new ConnectedComponentProcessor() {
				public void process(double[][] levels, List<int[]> connectedComponent) {
					if (connectedComponent.size() < 20 || connectedComponent.size() > 1000) {
						for (int[] pixel : connectedComponent) {
							levels[pixel[0]][pixel[1]] = 255.0;
						}
					}
				}
			};
			ImageUtils.processConnectedComponents(levels, 127.0, ccprocSmall);
			List<double[][]> lines = extractLines(levels);
			for (double[][] line : lines) {
				ImageUtils.display(ImageUtils.makeImage(line));
			}
		}
	}
	
}
