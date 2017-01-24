package edu.berkeley.cs.nlp.ocular.data;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import tberg.murphy.fileio.f;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class TextAndLineImagesLoader {
	
	public static class TextAndLineImagesDocument implements Document {
		private final String imgPathPrefix;
		private final String imgNameSuffix;
		private final String textPath;
		private final boolean useLongS;
		private final int numLines;
		private final int lineHeight;

		public TextAndLineImagesDocument(String imgPathPrefix, String imgNameSuffix, String textPath, boolean useLongS, int numLines, int lineHeight) {
			this.imgPathPrefix = imgPathPrefix;
			this.imgNameSuffix = imgNameSuffix;
			this.textPath = textPath;
			this.useLongS = useLongS;
			this.numLines = numLines;
			this.lineHeight = lineHeight;
		}

		public PixelType[][][] loadLineImages() {
			final PixelType[][][] observations = new PixelType[numLines][][];
			for (int i=0; i<numLines; ++i) {
				try {
					if (lineHeight >= 0) {
						observations[i] = ImageUtils.getPixelTypes(ImageUtils.resampleImage(f.readImage(imgPathPrefix + i + imgNameSuffix), lineHeight));
					} else {
						observations[i] = ImageUtils.getPixelTypes(f.readImage(imgPathPrefix + i + imgNameSuffix));
					}
				} catch (Exception e) {
					throw new RuntimeException("Couldn't read doc from: " + imgPathPrefix + i + imgNameSuffix);
				}
			}
			return observations;
		}

		public String[][] loadDiplomaticTextLines() {
			File textFile = new File(textPath);
			String[][] text = (!textFile.exists() ? null : f.readDocumentByCharacter(textPath, numLines));
			return text;
		}
		
		public String[][] loadNormalizedTextLines() {
			return null;
		}
		
		public List<String> loadNormalizedText() {
			return null;
		}
		
		public String baseName() {
			String[] split = imgPathPrefix.split("/");
			String baseNamePlusHyphen = split[split.length-1];
			return baseNamePlusHyphen.substring(0, baseNamePlusHyphen.length()-1);
		}

		public boolean useLongS() {
			return useLongS;
		}
	}
	
	public static List<Document> loadDocuments(String inputPath, int lineHeight) {
		List<String> lines = f.readLines(inputPath);
		List<Document> docs = new ArrayList<Document>();
		File inputFile = new File(inputPath);
		for (String line : lines) {
			if (line.trim().equals("")) continue;
			String[] split = line.split("\\s+");
			docs.add(new TextAndLineImagesDocument(inputFile.getParentFile().getAbsolutePath()+"/"+split[0], split[1], inputFile.getParentFile().getAbsolutePath()+"/"+split[2], Boolean.parseBoolean(split[3]), Integer.parseInt(split[4]), lineHeight));
		}
		return docs;
	}

}
