package edu.berkeley.cs.nlp.ocular.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.berkeley.cs.nlp.ocular.data.textreader.BasicTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.ConvertLongSTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.ExplicitCharacterSetTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.RemoveDiacriticsTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel.LMType;
import fig.Option;
import fig.OptionsParser;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class LMTrainMain implements Runnable {
	
	@Option(gloss = "Output LM file path.")
	public static String lmPath = null;
	
	@Option(gloss = "Input corpus path.")
	public static String textPath = null;
	
	@Option(gloss = "Use separate character type for long s.")
	public static boolean insertLongS = false;
	
	@Option(gloss = "Remove diacritics?")
	public static boolean removeDiacritics = false;

	@Option(gloss = "A set of valid characters. If a character with a diacritic is found but not in this set, the diacritic will be dropped. Other excluded characters will simply be dropped.")
	public static Set<String> explicitCharacterSet = null;

	@Option(gloss = "Maximum number of lines to use from corpus.")
	public static int maxLines = 1000000;
	
	@Option(gloss = "LM character n-gram length.")
	public static int charN = 6;
	
	@Option(gloss = "Exponent on LM scores.")
	public static double power = 4.0;
	
	
	public static void main(String[] args) {
		LMTrainMain main = new LMTrainMain();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] {main});
		if (!parser.doParse(args)) System.exit(1);
		main.run();
	}

	public void run() {
		if (lmPath == null) throw new IllegalArgumentException("-lmPath not set");
		if (textPath == null) throw new IllegalArgumentException("-textPath not set");
		
		TextReader textReader = new BasicTextReader();
		if (explicitCharacterSet != null) textReader = new ExplicitCharacterSetTextReader(textReader, explicitCharacterSet);
		if(removeDiacritics) textReader = new RemoveDiacriticsTextReader(textReader);
		if(insertLongS) textReader = new ConvertLongSTextReader(textReader);

		NgramLanguageModel lm = NgramLanguageModel.buildFromText(textPath, maxLines, charN, LMType.KNESER_NEY, power, textReader);
		writeLM(lm, lmPath);
	}
	
	public static NgramLanguageModel readLM(String lmPath) {
		NgramLanguageModel lm = null;
		try {
			File file = new File(lmPath);
			if (!file.exists()) {
				System.out.println("Serialized lm file " + lmPath + " not found");
				return null;
			}
			FileInputStream fileIn = new FileInputStream(file);
			ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(fileIn));
			lm = (NgramLanguageModel) in.readObject();
			in.close();
			fileIn.close();
		} catch(Exception e) {
			throw new RuntimeException(e);
		}
		return lm;
	}

	public static void writeLM(NgramLanguageModel lm, String lmPath) {
		try {
      new File(lmPath).getParentFile().mkdirs();
			FileOutputStream fileOut = new FileOutputStream(lmPath);
			ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(fileOut));
			out.writeObject(lm);
			out.close();
			fileOut.close();
		} catch(IOException e) {
			throw new RuntimeException(e);
		}
	}
	
}
