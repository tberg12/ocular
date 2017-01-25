package edu.berkeley.cs.nlp.ocular.data;

import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.ConnectedComponentProcessor;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.preprocessing.Binarizer;
import edu.berkeley.cs.nlp.ocular.preprocessing.LineExtractor;
import tberg.murphy.arrays.a;
import tberg.murphy.fileio.f;
import tberg.murphy.threading.BetterThreader;

public class FirstFolioRawImageLoader {

	public static class FirstFolioRawImageDocument implements Document {
		private final String baseName;
		final PixelType[][][] observations;
		
		public FirstFolioRawImageDocument(String inputPath, String baseName, int lineHeight, double binarizeThreshold) {
			this.baseName = baseName;
			double[][] levels = ImageUtils.getLevels(f.readImage(inputPath+"/"+baseName));
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
			Binarizer.binarizeGlobal(binarizeThreshold, levels);
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
			
			int padHeight = 0;
			double[][] topPadLevels = new double[levels.length][];
			for (int i=0; i<levels.length; ++i) topPadLevels[i] = a.append(a.add(a.zerosDouble(padHeight), 255.0), levels[i]);
			
			List<double[][]> lines = LineExtractor.extractLines(topPadLevels);
			observations = new PixelType[lines.size()][][];
			for (int i=0; i<lines.size(); ++i) {
				if (lineHeight >= 0) {
					observations[i] = ImageUtils.getPixelTypes(ImageUtils.resampleImage(ImageUtils.makeImage(lines.get(i)), lineHeight));
				} else {
					observations[i] = ImageUtils.getPixelTypes(ImageUtils.makeImage(lines.get(i)));
				}
			}
		}

		public PixelType[][][] loadLineImages() {
			return observations;
		}

		public String[][] loadDiplomaticTextLines() {
			return null;
		}
		
		public String[][] loadNormalizedTextLines() {
			return null;
		}
		
		public List<String> loadNormalizedText() {
			return null;
		}
		
		public String baseName() {
			return baseName;
		}
	}
	
	public static List<Document> loadDocuments(final String inputPath, final int lineHeight, final double binarizeThreshold, final int numThreads) {
		System.out.println("Extracting text line images from dataset "+inputPath);
		File dir = new File(inputPath);
		final String[] dirList = dir.list(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				if (name.startsWith(".")) { // ignore hidden files
					return false;
				}
				else if (!name.endsWith(".png") && !name.endsWith(".jpg")) {
					return false;
				}
				return true;
			}
		});
		final Document[] docs = new Document[dirList.length]; 
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer i, Object ignore){
			String baseName = dirList[i];
			docs[i] = new FirstFolioRawImageDocument(inputPath, baseName, lineHeight, binarizeThreshold);
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int i=0; i<dirList.length; ++i) threader.addFunctionArgument(i);
		threader.run();
		return Arrays.asList(docs);
	}

}
