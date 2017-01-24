package edu.berkeley.cs.nlp.ocular.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.image.FontRenderer;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.lm.LanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import tberg.murphy.fig.Option;
import tberg.murphy.fileio.f;
import tberg.murphy.indexer.Indexer;
import tberg.murphy.threading.BetterThreader;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class InitializeFont extends OcularRunnable {

	@Option(gloss = "Path to the language model file (so that it knows which characters to create images for).")
	public static String inputLmPath = null; // Required.

	@Option(gloss = "Output font file path.")
	public static String outputFontPath = null; // Required.
	
	@Option(gloss = "Path to a file that contains a custom list of font names that may be used to initialize the font. The file should contain one font name per line. Default: Use all valid fonts found on the computer.")
	public static String allowedFontsPath = null;
	
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
		System.out.println("InitializeFont");
		InitializeFont main = new InitializeFont();
		main.doMain(main, args);
	}
	
	protected void validateOptions() {
		if (inputLmPath == null) throw new IllegalArgumentException("-inputLmPath not set");
		if (outputFontPath == null) throw new IllegalArgumentException("-outputFontPath not set");
	}

	public void run(List<String> commandLineArgs) {
		Set<String> allowedFonts = getAllowedFontsListFromFile();
		
		final LanguageModel lm = InitializeLanguageModel.readLM(inputLmPath);
		final Indexer<String> charIndexer = lm.getCharacterIndexer();
		final CharacterTemplate[] templates = new CharacterTemplate[charIndexer.size()];
		final PixelType[][][][] fontPixelData = FontRenderer.getRenderedFont(charIndexer, CharacterTemplate.LINE_HEIGHT, allowedFonts);
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
		Map<String,CharacterTemplate> charTemplates = new HashMap<String, CharacterTemplate>();
		for (CharacterTemplate template : templates) {
			charTemplates.put(template.getCharacter(), template);
		}
		System.out.println("Writing intialized font to" + outputFontPath);
		InitializeFont.writeFont(new Font(charTemplates), outputFontPath);
	}

	private Set<String> getAllowedFontsListFromFile() {
		Set<String> allowedFonts = new HashSet<String>();
		if (allowedFontsPath != null) {
			for (String fontName : f.readLines(allowedFontsPath)) {
				allowedFonts.add(fontName);
			}
		}
		return allowedFonts;
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
	public static Font readFont(String fontPath) {
		ObjectInputStream in = null;
		try {
			File file = new File(fontPath);
			if (!file.exists()) {
				throw new RuntimeException("Serialized font file " + fontPath + " not found");
			}
			in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(file)));
			Object obj = in.readObject();

			{ // TODO: For legacy font models...
				if (obj instanceof Map<?, ?>) 
					return new Font((Map<String, CharacterTemplate>)obj);
			}
			
			return (Font) obj;
		} catch (Exception e) {
			throw new RuntimeException(e);
		} finally {
			if (in != null)
				try { in.close(); } catch (IOException e) { throw new RuntimeException(e); }
		}
	}

	public static void writeFont(Font font, String fontPath) {
		ObjectOutputStream out = null;
		try {
			new File(fontPath).getAbsoluteFile().getParentFile().mkdirs();
			out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(fontPath)));
			out.writeObject(font);
		} catch (Exception e) {
			throw new RuntimeException(e);
		} finally {
			if (out != null)
				try { out.close(); } catch (IOException e) { throw new RuntimeException(e); }
		}
	}

}
