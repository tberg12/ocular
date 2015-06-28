package edu.berkeley.cs.nlp.ocular.data;

import fileio.f;
import edu.berkeley.cs.nlp.ocular.data.textreader.BasicTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.preprocessing.Binarizer;
import edu.berkeley.cs.nlp.ocular.preprocessing.Cropper;
import edu.berkeley.cs.nlp.ocular.preprocessing.LineExtractor;
import edu.berkeley.cs.nlp.ocular.preprocessing.Straightener;

public class LazyRawImageLoader implements ImageLoader {

	public static class RawImageDocument implements ImageLoader.Document {
		private final String inputPath;
		private final String baseName;
		private final int lineHeight;
		private final double binarizeThreshold;
		private final boolean crop;

		private PixelType[][][] observations = null;
		private String[][] text = null;
		
		private TextReader textReader = new BasicTextReader();

		public RawImageDocument(String inputPath, String baseName, int lineHeight, double binarizeThreshold, boolean crop) {
			this.inputPath = inputPath;
			this.baseName = baseName;
			this.lineHeight = lineHeight;
			this.binarizeThreshold = binarizeThreshold;
			this.crop = crop;
		}

		public PixelType[][][] loadLineImages() {
			if (observations == null) {
				String fn = inputPath + "/" + baseName;
				System.out.println("Extracting text line images from " + fn);
				double[][] levels = ImageUtils.getLevels(f.readImage(fn));
				double[][] rotLevels = Straightener.straighten(levels);
				double[][] cropLevels = crop ? Cropper.crop(rotLevels, binarizeThreshold) : rotLevels;
				Binarizer.binarizeGlobal(binarizeThreshold, cropLevels);
				List<double[][]> lines = LineExtractor.extractLines(cropLevels);
				observations = new PixelType[lines.size()][][];
				for (int i = 0; i < lines.size(); ++i) {
					if (lineHeight >= 0) {
						observations[i] = ImageUtils.getPixelTypes(ImageUtils.resampleImage(ImageUtils.makeImage(lines.get(i)), lineHeight));
					}
					else {
						observations[i] = ImageUtils.getPixelTypes(ImageUtils.makeImage(lines.get(i)));
					}
				}
			}
			return observations;
		}

		public String[][] loadLineText() {
			if (text == null) {
				File textFile = new File(inputPath, baseName.replaceAll("\\.[^.]*$", ".txt"));
				if (textFile.exists()) {
					System.out.println("Evaluation text found at " + textFile);
					List<List<String>> textList = new ArrayList<List<String>>();
					try {
						BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(textFile), "UTF-8"));
						while (in.ready()) {
							textList.add(textReader.readCharacters(in.readLine()));
						}
						in.close();
					}
					catch (IOException e) {
						throw new RuntimeException(e);
					}

					text = new String[textList.size()][];
					for (int i = 0; i < text.length; ++i) {
						List<String> line = textList.get(i);
						text[i] = line.toArray(new String[line.size()]);
					}
				}
				else {
					System.out.println("No evaluation text found at " + textFile);
				}
			}
			return text;
		}

		public String baseName() {
			return baseName;
		}

		public boolean useLongS() {
			return false;
		}
		
	}

	private final String inputPath;
	private final int lineHeight;
	private final double binarizeThreshold;
	private final boolean crop;

	public LazyRawImageLoader(String inputPath, int lineHeight, double binarizeThreshold, boolean crop) {
		this.inputPath = inputPath;
		this.lineHeight = lineHeight;
		this.binarizeThreshold = binarizeThreshold;
		this.crop = crop;
	}

	public List<Document> readDataset() {
		File dir = new File(inputPath);
		System.out.println("Reading data from ["+dir+ "], which "+(dir.exists() ? "exists" : "does not exist"));
		String[] dirList = dir.list(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				return !name.startsWith(".") && !name.endsWith(".txt") && !(new File(dir, name).isDirectory());
			}
		});
		List<Document> docs = new ArrayList<Document>();
		for (String baseName : dirList) {
			docs.add(new RawImageDocument(inputPath, baseName, lineHeight, binarizeThreshold, crop));
		}
		return docs;
	}
}
