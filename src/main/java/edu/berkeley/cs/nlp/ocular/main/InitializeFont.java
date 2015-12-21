package edu.berkeley.cs.nlp.ocular.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.image.FontRenderer;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.lm.LanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import fig.Option;
import fig.OptionsParser;
import indexer.Indexer;
import threading.BetterThreader;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class InitializeFont implements Runnable {

	@Option(gloss = "Path to the language model file (so that it knows which characters to create images for).")
	public static String lmPath = null;

	@Option(gloss = "Output font file path.")
	public static String fontPath = null;

	@Option(gloss = "Number of threads to use.")
	public static int numFontInitThreads = 8;
	
	@Option(gloss = "Max template width as fraction of text line height.")
	public static double templateMaxWidthFraction = 1.0;

	@Option(gloss = "Min template width as fraction of text line height.")
	public static double templateMinWidthFraction = 0.0;

	@Option(gloss = "Max space template width as fraction of text line height.")
	public static double spaceMaxWidthFraction = 1.0;

	@Option(gloss = "Min space template width as fraction of text line height.")
	public static double spaceMinWidthFraction = 0.0;
	
	
	public static void main(String[] args) {
		InitializeFont main = new InitializeFont();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] {main});
		if (!parser.doParse(args)) System.exit(1);
		main.run();
	}

	public void run() {
		if (lmPath == null) throw new IllegalArgumentException("-lmPath not set");
		if (fontPath == null) throw new IllegalArgumentException("-fontPath not set");
		
		final LanguageModel lm = readLM(lmPath);
		final Indexer<String> charIndexer = lm.getCharacterIndexer();
		final CharacterTemplate[] templates = new CharacterTemplate[charIndexer.size()];
		final PixelType[][][][] fontPixelData = FontRenderer.getRenderedFont(charIndexer, CharacterTemplate.LINE_HEIGHT);
//		final PixelType[][][] fAndBarFontPixelData = buildFAndBarFontPixelData(charIndexer, fontPixelData);
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer c, Object ignore){
			String currChar = charIndexer.getObject(c);
			if (!currChar.equals(Charset.SPACE)) {
				templates[c] = new CharacterTemplate(currChar, (float) templateMaxWidthFraction, (float) templateMinWidthFraction);
//				if (currChar.equals(Charset.LONG_S)) {
//					templates[c].initializeAndSetPriorFromFontData(fAndBarFontPixelData);
//				} else {
					templates[c].initializeAndSetPriorFromFontData(fontPixelData[c]);
//				}
			} else {
				templates[c] = new CharacterTemplate(Charset.SPACE, (float) spaceMaxWidthFraction, (float) spaceMinWidthFraction);
			}
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numFontInitThreads);
		for (int c=0; c<templates.length; ++c) threader.addFunctionArgument(c);
		threader.run();
		Map<String,CharacterTemplate> font = new HashMap<String, CharacterTemplate>();
		for (CharacterTemplate template : templates) {
			font.put(template.getCharacter(), template);
		}
		InitializeFont.writeFont(font, fontPath);
	}
	
	public static LanguageModel readLM(String lmPath) {
		LanguageModel lm = null;
		try {
			File file = new File(lmPath);
			if (!file.exists()) {
				System.out.println("Serialized lm file " + lmPath + " not found");
				return null;
			}
			FileInputStream fileIn = new FileInputStream(file);
			ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(fileIn));
			lm = (LanguageModel) in.readObject();
			in.close();
			fileIn.close();
		} catch(Exception e) {
			throw new RuntimeException(e);
		}
		return lm;
	}

//	private static PixelType[][][] buildFAndBarFontPixelData(Indexer<String> charIndexer, PixelType[][][][] fontPixelData) {
//		List<PixelType[][]> fAndBarFontPixelData = new ArrayList<PixelType[][]>();
//		if (charIndexer.contains("f")) {
//			int c = charIndexer.getIndex("f");
//			for (PixelType[][] datum : fontPixelData[c]) {
//				fAndBarFontPixelData.add(datum);
//			}
//		}
//		if (charIndexer.contains("|")) {
//			int c = charIndexer.getIndex("|");
//			for (PixelType[][] datum : fontPixelData[c]) {
//				fAndBarFontPixelData.add(datum);
//			}
//		}
//		return fAndBarFontPixelData.toArray(new PixelType[0][][]);
//	}
	
	@SuppressWarnings("unchecked")
	public static Map<String,CharacterTemplate> readFont(String fontPath) {
		Map<String,CharacterTemplate> font = null;
		try {
			File file = new File(fontPath);
			if (!file.exists()) {
				System.out.println("Serialized font file " + fontPath + " not found");
				return null;
			}
			FileInputStream fileIn = new FileInputStream(file);
			ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(fileIn));
			font = (Map<String,CharacterTemplate>) in.readObject();
			in.close();
			fileIn.close();
		} catch(Exception e) {
			throw new RuntimeException(e);
		}
		return font;
	}

	public static void writeFont(Map<String,CharacterTemplate> font, String fontPath) {
		try {
			new File(fontPath).getParentFile().mkdirs();
			FileOutputStream fileOut = new FileOutputStream(fontPath);
			ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(fileOut));
			out.writeObject(font);
			out.close();
			fileOut.close();
		} catch(IOException e) {
			throw new RuntimeException(e);
		}
	}

}
