package edu.berkeley.cs.nlp.ocular.preprocessing;

import java.io.File;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import fileio.f;

public class Test {

	public static void main(String[] args) {
		String path = "/Users/tberg/Dropbox/corpora/ocr_data/autocrop_dev/";
		File dir = new File(path);
		for (String name : dir.list()) {
			double[][] levels = ImageUtils.getLevels(f.readImage(path+"/"+name));
			double[][] rotLevels = Straightener.straighten(levels);
			Binarizer.binarizeGlobal(0.08, rotLevels);
			ImageUtils.display(ImageUtils.makeImage(rotLevels));
			
			
//			double[][] cropLevels = Cropper.crop(rotLevels);
//			ImageUtils.display(ImageUtils.makeImage(cropLevels));
		}
	}

}
