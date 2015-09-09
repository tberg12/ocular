package edu.berkeley.cs.nlp.ocular.data;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.textreader.BasicTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import fileio.f;

/**
 * A document that reads a file only as it is needed (and then stores
 * the contents in memory for later use).
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class LazyRawSingleImageDocument extends LazyRawImageDocument {
	private final File file;

	private String[][] text = null;

	private TextReader textReader = new BasicTextReader();

	public LazyRawSingleImageDocument(File file, String inputPath, int lineHeight, double binarizeThreshold, boolean crop, String preextractedLinesPath) {
		super(inputPath, lineHeight, binarizeThreshold, crop, preextractedLinesPath);
		this.file = file;
	}

	protected BufferedImage doLoadBufferedImage() {
		System.out.println("Extracting text line images from " + file);
		return f.readImage(file.getPath());
  }
	
	protected File file() { return file; }
	protected String preext() { return FileUtil.withoutExtension(file.getName()); }
	protected String ext() { return FileUtil.extension(file.getName()); }

	public String[][] loadLineText() {
		if (text == null) {
			File textFile = new File(file.getPath().replaceAll("\\.[^.]*$", ".txt"));
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
				System.out.println("No evaluation text found at " + textFile + ".  (This isn't necessarily a problem.)");
			}
		}
		return text;
	}

	public String baseName() {
		return file.getPath();
	}

}
