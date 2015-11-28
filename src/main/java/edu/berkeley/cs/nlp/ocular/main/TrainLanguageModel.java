package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;
import indexer.HashMapIndexer;
import indexer.Indexer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.berkeley.cs.nlp.ocular.data.FileUtil;
import edu.berkeley.cs.nlp.ocular.data.textreader.BasicTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.data.textreader.ConvertLongSTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.ExplicitCharacterSetTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.RemoveDiacriticsTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.ReplaceSomeTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CorpusCounter;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel.LMType;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import fig.Option;
import fig.OptionsParser;
import fileio.f;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class TrainLanguageModel implements Runnable {
	
	@Option(gloss = "Output LM file path.")
	public static String lmPath = null;
	
	@Option(gloss = "Path to the text files (or directory hierarchies) for training the LM.  For each entry, the entire directory will be recursively searched for any files that do not start with `.`.  For a multilingual (code-switching) model, give multiple comma-separated files with language names: \"english->texts/english/,spanish->texts/spanish/,french->texts/french/\".  If spaces are used, be sure to wrap the whole string with \"quotes\".)")
	public static String textPath = null;
	
	@Option(gloss = "Prior probability of each language; ignore for uniform priors. Give multiple comma-separated language/prior pairs: \"english->0.7,spanish->0.2,french->0.1\". If spaces are used, be sure to wrap the whole string with \"quotes\".  Defaults to uniform priors. (Only relevant if multiple languages used.)")
	public static String languagePriors = null;
	
	@Option(gloss = "Prior probability of sticking with the same language when moving between words in a code-switch model transition model. (Only relevant if multiple languages used.)")
	public static double pKeepSameLanguage = 0.999999;

	@Option(gloss = "Paths to Alternate Spelling Replacement files. If just a simple path is given, the replacements will be applied to all languages.  For language-specific replacements, give multiple comma-separated language/path pairs: \"english->rules/en.txt,spanish->rules/sp.txt,french->rules/fr.txt\". If spaces are used, be sure to wrap the whole string with \"quotes\". Any languages for which no replacements are need can be safely ignored.")
	public static String alternateSpellingReplacementPaths = null;
	
	@Option(gloss = "Use separate character type for long s.")
	public static boolean insertLongS = false;
	
	@Option(gloss = "Remove diacritics?")
	public static boolean removeDiacritics = false;

	@Option(gloss = "A set of valid characters. If a character with a diacritic is found but not in this set, the diacritic will be dropped. Other excluded characters will simply be dropped. Ignore to allow all characters.")
	public static Set<String> explicitCharacterSet = null;

	@Option(gloss = "Maximum number of lines to use from corpus.")
	public static int maxLines = 1000000;
	
	@Option(gloss = "LM character n-gram length.")
	public static int charN = 6;
	
	@Option(gloss = "Exponent on LM scores.")
	public static double power = 4.0;
	
	@Option(gloss = "Number of characters to use for training the LM.  Use -1 to indicate that the full training data should be used.")
	public static long lmCharCount = -1;

	
	public static void main(String[] args) {
		TrainLanguageModel main = new TrainLanguageModel();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] { main });
		if (!parser.doParse(args)) System.exit(1);
		main.run();
	}

	public void run() {
		if (lmPath == null) throw new IllegalArgumentException("-lmPath not set");
		if (textPath == null) throw new IllegalArgumentException("-textPath not set");
		
		Map<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReadersAndPriors = makePathsReadersAndPriors();

		Indexer<String> charIndexer = new HashMapIndexer<String>();
		Map<String, Tuple2<SingleLanguageModel, Double>> lmsAndPriors = makeMultipleSubLMs(pathsReadersAndPriors, charIndexer);
		charIndexer.lock();

		System.out.println("pKeepSameLanguage = " + pKeepSameLanguage);
		double priorSum = 0.0;
		for(String language: lmsAndPriors.keySet())
			priorSum += lmsAndPriors.get(language)._2;
		StringBuilder priorsSb = new StringBuilder("Language priors: ");
		for(String language: lmsAndPriors.keySet())
			priorsSb.append(language).append(" -> ").append(lmsAndPriors.get(language)._2 / priorSum).append(", ");
		System.out.println(priorsSb.substring(0, priorsSb.length() - 2));
		System.out.println("charN = " + charN);

		CodeSwitchLanguageModel codeSwitchLM = new BasicCodeSwitchLanguageModel(lmsAndPriors, charIndexer, pKeepSameLanguage, charN);
		System.out.println("writing LM to " + lmPath);
		writeLM(codeSwitchLM, lmPath);
	}

	public Map<String, Tuple2<Tuple2<String, TextReader>, Double>> makePathsReadersAndPriors() {
		Map<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReadersAndPriors = new HashMap<String, Tuple2<Tuple2<String, TextReader>, Double>>();

		String textPathString = textPath;
		if (!textPath.contains("->")) textPathString = "NoLanguageNameGiven->" + textPath; // repair "invalid" input
		Map<String, String> languagePathMap = new HashMap<String, String>();
		for (String part : textPathString.split(",")) {
			String[] subparts = part.split("->");
			if (subparts.length != 2) throw new IllegalArgumentException("malformed lmPath argument: comma-separated part must be of the form \"LANGUAGE->PATH\", was: " + part);
			String language = subparts[0].trim();
			String filepath = subparts[1].trim();
			languagePathMap.put(language, filepath);
		}

		Map<String, Double> languagePriorMap = new HashMap<String, Double>();
		if (languagePriors != null) {
			for (String part : languagePriors.split(",")) {
				String[] subparts = part.split("->");
				if (subparts.length != 2) throw new IllegalArgumentException("malformed languagePriors argument: comma-separated part must be of the form \"LANGUAGE->PRIOR\", was: " + part);
				String language = subparts[0].trim();
				Double prior = Double.parseDouble(subparts[1].trim());
				languagePriorMap.put(language, prior);
			}
			if (!languagePathMap.keySet().equals(languagePriorMap.keySet()))
				throw new RuntimeException("-textPath and -languagePriors do not have the same set of languages: " + languagePathMap.keySet() + " vs " + languagePriorMap.keySet());
		}
		else {
			for (String language : languagePathMap.keySet())
				languagePriorMap.put(language, 1.0);
		}
		
		Map<String, String> languageAltSpellPathMap = new HashMap<String, String>();
		if (alternateSpellingReplacementPaths != null) {
			if (!alternateSpellingReplacementPaths.contains("->")) { // only one path, use for all languages
				String replacementsPath = alternateSpellingReplacementPaths;
				for (String language : languagePathMap.keySet()) {
					languageAltSpellPathMap.put(language, replacementsPath);
				}
			}
			else {
				for (String part : alternateSpellingReplacementPaths.split(",")) {
					String[] subparts = part.split("->");
					if (subparts.length != 2) throw new IllegalArgumentException("malformed alternateSpellingReplacementPaths argument: comma-separated part must be of the form \"LANGUAGE->PATH\", was: " + part);
					String language = subparts[0].trim();
					String replacementsPath = subparts[1].trim();
					if (!languagePathMap.keySet().contains(language)) throw new RuntimeException("Language '"+language+"' appears in the alternateSpellingReplacementPaths argument but not in textPath ("+languagePathMap.keySet()+")");
					languageAltSpellPathMap.put(language, replacementsPath);
				}
			}
		}
		
		for (String language : languagePathMap.keySet()) {
			String filepath = languagePathMap.get(language);
			Double prior = languagePriorMap.get(language);
			System.out.println("For language '" + language + "', using text in " + filepath + ", prior=" + prior
					+ (languageAltSpellPathMap.keySet().contains(language) ? ", alternate spelling replacement rules in " + languageAltSpellPathMap.get(language) : ""));
			
			TextReader textReader = new BasicTextReader();
			if (explicitCharacterSet != null) textReader = new ExplicitCharacterSetTextReader(textReader, explicitCharacterSet);
			if (removeDiacritics) textReader = new RemoveDiacriticsTextReader(textReader);
			if (insertLongS) textReader = new ConvertLongSTextReader(textReader);
			if (languageAltSpellPathMap.keySet().contains(language)) textReader = handleReplacementRulesOption(textReader, languageAltSpellPathMap.get(language));
			
			pathsReadersAndPriors.put(language, makeTuple2(makeTuple2(filepath, textReader), prior));
		}

		return pathsReadersAndPriors;
	}
	
	private TextReader handleReplacementRulesOption(TextReader textReader, String replacementsFilePath) {
		File replacementsFile = new File(replacementsFilePath);
		if (!replacementsFile.exists()) throw new RuntimeException("replacementsFile [" + replacementsFilePath + "] does not exist");
		List<Tuple2<Tuple2<List<String>, List<String>>, Integer>> rules = ReplaceSomeTextReader.loadRulesFromFile(replacementsFilePath);
		for (Tuple2<Tuple2<List<String>, List<String>>, Integer> rule : rules)
			System.out.println("    " + rule);
		return new ReplaceSomeTextReader(rules, textReader);
	}

	private Map<String, Tuple2<SingleLanguageModel, Double>> makeMultipleSubLMs(Map<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReadersAndPriors, Indexer<String> charIndexer) {
		Map<String, Tuple2<SingleLanguageModel, Double>> lmsAndPriors = new HashMap<String, Tuple2<SingleLanguageModel, Double>>();
		for (Map.Entry<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReaderAndPrior : pathsReadersAndPriors.entrySet()) {
			String language = pathsReaderAndPrior.getKey();
			String filepath = pathsReaderAndPrior.getValue()._1._1;
			TextReader textReader = pathsReaderAndPrior.getValue()._1._2;
			System.out.println(language + " text reader: " + textReader);

			CorpusCounter counter = new CorpusCounter(charN);
			List<String> chars = readFileChars(filepath, textReader, lmCharCount > 0 ? lmCharCount : Long.MAX_VALUE);
			System.out.println("  using " + chars.size() + " characters for " + language + " read from " + filepath);
			counter.countChars(chars, charIndexer, 0);

			Double prior = pathsReaderAndPrior.getValue()._2;

			List<String> langChars = new ArrayList<String>();
			for (int i : counter.getActiveCharacters())
				langChars.add(charIndexer.getObject(i));
			Collections.sort(langChars);
			System.out.println(language + ": " + langChars);

			SingleLanguageModel lm = new NgramLanguageModel(charIndexer, counter.getCounts(), counter.getActiveCharacters(), LMType.KNESER_NEY, power);
			lmsAndPriors.put(language, makeTuple2(lm, prior));
		}
		charIndexer.lock();
		return lmsAndPriors;
	}

	private List<String> readFileChars(String filepath, TextReader textReader, long charsToTake) {
		List<String> allChars = new ArrayList<String>();
		outer: 
			for (File file : FileUtil.recursiveFiles(filepath)) {
				for (String line : f.readLines(file.getPath())) {
					if (allChars.size() >= charsToTake) break outer;
					List<String> chars = textReader.readCharacters(line + " ");
					
					for (String c: chars) {
						Charset.escapeChar(c);
						Charset.unescapeChar(c);
					}
					
					allChars.addAll(chars);
				}
			}
		return allChars;
	}
	
	public static CodeSwitchLanguageModel readLM(String lmPath) {
		CodeSwitchLanguageModel lm = null;
		try {
			File file = new File(lmPath);
			if (!file.exists()) {
				throw new RuntimeException("Serialized CodeSwitchLanguageModel file " + lmPath + " not found");
			}
			FileInputStream fileIn = new FileInputStream(file);
			ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(fileIn));
			lm = (CodeSwitchLanguageModel) in.readObject();
			in.close();
			fileIn.close();
		} catch(Exception e) {
			throw new RuntimeException(e);
		}
		return lm;
	}

	public static void writeLM(CodeSwitchLanguageModel lm, String lmPath) {
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
