package edu.berkeley.cs.nlp.ocular.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.textreader.BasicTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import fileio.f;

/**
 * Read a dataset that is pre-split by line across multiple files.
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class SplitLineImageLoader {//implements DatasetLoader {

	public static class SplitLineImageDocument implements ImageLoader.Document {
		private final String[] lineImagePaths;
		private final String baseName;
		private final int lineHeight;
		private PixelType[][][] observations = null;
		private String[][] text = null;

		private TextReader textReader = new BasicTextReader();

		public SplitLineImageDocument(String[] lineImagePaths, String baseName, int lineHeight) {
			this.lineImagePaths = lineImagePaths;
			this.baseName = baseName;
			this.lineHeight = lineHeight;

			for (String lineImagePath : lineImagePaths)
				if (!new File(lineImagePath).exists()) throw new IllegalArgumentException("lineImageFile does not exists: " + lineImagePath);
		}

		public PixelType[][][] loadLineImages() {
			if (observations == null) {
				observations = new PixelType[lineImagePaths.length][][];
				for (int i = 0; i < lineImagePaths.length; ++i) {
					String lineImageFile = lineImagePaths[i];
					System.out.println("Loading pre-extracted line from " + lineImageFile);
					try {
						if (lineHeight >= 0) {
							observations[i] = ImageUtils.getPixelTypes(ImageUtils.resampleImage(f.readImage(lineImageFile), lineHeight));
						}
						else {
							observations[i] = ImageUtils.getPixelTypes(f.readImage(lineImageFile));
						}
					}
					catch (Exception e) {
						throw new RuntimeException("Couldn't read line image from: " + lineImageFile);
					}
				}
				//f.writeImage("/Users/dhg/Desktop/"+(new File(lineImagePaths[0]).getName()), Visualizer.renderLineExtraction(observations));
			}
			return observations;
		}

		public String[][] loadLineText() {
			if (text == null) {
				List<List<String>> textList = new ArrayList<List<String>>();
				for (String lineImageFile : lineImagePaths) {
					File textFile = new File(lineImageFile.replaceAll("\\.[^.]*$", ".txt"));
					if (textFile.exists()) {
						System.out.println("Loading pre-extracted text from " + textFile);
						List<String> buffer = new ArrayList<String>();
						try {
							BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(textFile), "UTF-8"));
							while (in.ready()) {
								buffer.addAll(textReader.readCharacters(in.readLine()));
							}
							in.close();
						}
						catch (IOException e) {
							throw new RuntimeException(e);
						}
						textList.add(buffer);
					}
					else {
						System.out.println("No evaluation text found at " + textFile);
						break;
					}
				}

				if (!textList.isEmpty()) {
					text = new String[textList.size()][];
					for (int i = 0; i < text.length; ++i) {
						List<String> line = textList.get(i);
						text[i] = line.toArray(new String[line.size()]);
						System.out.println((i + 1) + ": " + StringHelper.join(text[i]));
					}
				}
			}
			return text;
		}

		public String baseName() {
			return baseName;
		}

	}

}
