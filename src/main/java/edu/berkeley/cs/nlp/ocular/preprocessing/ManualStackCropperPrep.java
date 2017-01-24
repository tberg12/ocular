package edu.berkeley.cs.nlp.ocular.preprocessing;

import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.ConnectedComponentProcessor;
import tberg.murphy.fileio.f;

public class ManualStackCropperPrep {
	
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
		File oddDirCol1 = new File(path + "/odd_col1");
		File oddDirCol2 = new File(path + "/odd_col2");
		oddDirCol1.mkdirs();
		oddDirCol2.mkdirs();
		File evenDirCol1 = new File(path + "/even_col1");
		File evenDirCol2 = new File(path + "/even_col2");
		evenDirCol1.mkdirs();
		evenDirCol2.mkdirs();
		File dirExtr = new File(path + "/col_extraction");
		dirExtr.mkdirs();
		boolean odd = false;
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
			f.writeImage((odd ? oddDirCol1.getAbsolutePath() : evenDirCol1.getAbsolutePath()) +"/"+ baseName + "_col1.png", ImageUtils.makeImage(rotLevels));
			f.writeImage((odd ? oddDirCol2.getAbsolutePath() : evenDirCol2.getAbsolutePath()) +"/"+ baseName + "_col2.png", ImageUtils.makeImage(rotLevels));
			odd = !odd;
		}
	}

}
