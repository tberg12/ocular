package edu.berkeley.cs.nlp.ocular.preprocessing;

import java.awt.image.BufferedImage;
import java.io.File;

import edu.berkeley.cs.nlp.ocular.data.PdfImageReader;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import fileio.f;

public class Test {

	public static void main(String[] args) {
//		String path = "sample_images/multilingual/";
//		String path = "/Users/dhg/Desktop/pl/";
//		File dir = new File(path);
//		for (String name : dir.list()) {
//			double[][] levels = ImageUtils.getLevels(f.readImage(path+"/"+name));
//			double[][] rotLevels = Straightener.straighten(levels);
//			Binarizer.binarizeGlobal(0.08, rotLevels);
//			ImageUtils.display(ImageUtils.makeImage(rotLevels));
//			
//			
////			double[][] cropLevels = Cropper.crop(rotLevels);
////			ImageUtils.display(ImageUtils.makeImage(cropLevels));

		
		
		
		
		String path = "/Users/dhg/Desktop/pl/";
		File dir = new File(path);
		for (String name : dir.list()) {
			for (BufferedImage image : PdfImageReader.readPdfAsImages(new File(path + "/" + name))) {
				double[][] levels = ImageUtils.getLevels(image);
				double[][] rotLevels = Straightener.straighten(levels);
				Binarizer.binarizeGlobal(0.08, rotLevels);
				ImageUtils.display(ImageUtils.makeImage(rotLevels));
			}

		}
	}

}
