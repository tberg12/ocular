package edu.berkeley.cs.nlp.ocular.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.berkeley.cs.nlp.ocular.data.textreader.BasicTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.ConvertLongSTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.WhitelistCharacterSetTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.FlipUVTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.RemoveAllDiacriticsTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel.LMType;
import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import tberg.murphy.fig.Option;
import tberg.murphy.fig.OptionsParser;

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
	public static boolean insertLongS = true;
	
	@Option(gloss = "Allow 'u' and 'v' to interchange.")
	public static boolean allowUVFlip = true;

	@Option(gloss = "Remove diacritics?")
	public static boolean removeDiacritics = false;

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
		
		String HYPHEN = "-";
		Set<String> PUNC = CollectionHelper.makeSet("&", ".", ",", ";", ":", "\"", "'", "!", "?", "(", ")", HYPHEN); 
		Set<String> ALPHABET = CollectionHelper.makeSet("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"); 
		
		Set<String> explicitCharacterSet = new HashSet<String>();
		explicitCharacterSet.addAll(PUNC);
		explicitCharacterSet.addAll(ALPHABET);
		explicitCharacterSet.add(HYPHEN);

		TextReader textReader = new BasicTextReader(false);
		textReader = new WhitelistCharacterSetTextReader(explicitCharacterSet, textReader);
		if(removeDiacritics) textReader = new RemoveAllDiacriticsTextReader(textReader);
		if(insertLongS) textReader = new ConvertLongSTextReader(textReader);
		if(allowUVFlip) textReader = new FlipUVTextReader(0.5, textReader);

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
			new File(lmPath).getAbsoluteFile().getParentFile().mkdirs();
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
