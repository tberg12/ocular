package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.makeTuple3;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.berkeley.cs.nlp.ocular.data.FileUtil;
import edu.berkeley.cs.nlp.ocular.data.textreader.BasicTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.CharsetHelper;
import edu.berkeley.cs.nlp.ocular.data.textreader.ConvertLongSTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.RemoveDiacriticsTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.ReplaceSomeTextReader;
import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CorpusCounter;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel.LMType;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.UniformLanguageModel;
import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import fig.Option;
import fig.OptionsParser;
import fileio.f;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class CodeSwitchLMTrainMain implements Runnable {
	
	@Option(gloss = "Output LM file path.")
	public static String lmPath = "lm/my_lm.lmser";
	
	@Option(gloss = "Make a code-switching model?")
	public static boolean codeSwitch = true;

	@Option(gloss = "Prior probability of sticking with the same language when moving between words in a code-switch model transition model.  (For use with codeSwitch.)")
	public static double pKeepSameLanguage = 0.999999;

	@Option(gloss = "Path to the text files for training the LM. (For multiple paths for multilingual (code-switching) support, give multiple comma-separated files with language names: \"english->lms/english.lmser,spanish->lms/spanish.lmser,french->lms/french.lmser\".  If spaces are used, be sure to wrap the whole string with \"quotes\".")
	public static String textPaths = "texts/test.txt";
	
	@Option(gloss = "Prior probability of each language; ignore for uniform priors. Give multiple comma-separated language, prior pairs: \"english->0.7,spanish->0.2,french->0.1\". If spaces are used, be sure to wrap the whole string with \"quotes\".")
	public static String languagePriors = null;
	
	@Option(gloss = "Paths to Alternate Spelling Replacement files. Give multiple comma-separated language, path pairs: \"english->rules/en.txt,spanish->rules/sp.txt,french->rules/fr.txt\". If spaces are used, be sure to wrap the whole string with \"quotes\". Any languages for which no replacements are need can be safely ignored.")
	public static String alternateSpellingReplacementPaths = null;
	
	@Option(gloss = "Use separate character type for long s.")
	public static boolean useLongS = false;
	
	@Option(gloss = "Keep diacritics?")
	public static boolean keepDiacritics = true;

	@Option(gloss = "Maximum number of lines to use from corpus.")
	public static int maxLines = 1000000;
	
	@Option(gloss = "LM character n-gram length.")
	public static int charN = 6;
	
	@Option(gloss = "Exponent on LM scores.")
	public static double power = 4.0;
	
	@Option(gloss = "Number of characters to use for training the LM.  Use -1 to indicate that the full training data should be used.")
	public static long lmCharCount = -1;

	@Option(gloss = "Ignore the training data and just return a uniform model?")
	public static boolean uniform = false;

	public static void main(String[] args) {
		CodeSwitchLMTrainMain main = new CodeSwitchLMTrainMain();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] {main});
		if (!parser.doParse(args)) System.exit(1);
		main.run();
	}

	public void run() {
		Map<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReadersAndPriors = makePathsReadersAndPriors();

		Map<String, Tuple3<SingleLanguageModel, Set<String>, Double>> lmsAndPriors;
		Indexer<String> charIndexer = new HashMapIndexer<String>();
		if (codeSwitch)
			lmsAndPriors = makeMultipleSubLMs(pathsReadersAndPriors, charIndexer);
		else
			lmsAndPriors = makeSingleSubLM(pathsReadersAndPriors, charIndexer);
		charIndexer.lock();

		System.out.println("pKeepSameLanguage = " + pKeepSameLanguage);
		System.out.println("charN = " + charN);

		CodeSwitchLanguageModel codeSwitchLM = new BasicCodeSwitchLanguageModel(lmsAndPriors, charIndexer, pKeepSameLanguage, charN);
		System.out.println("writing LM to " + lmPath);
		writeLM(codeSwitchLM, lmPath);
	}

	public Map<String, Tuple2<Tuple2<String, TextReader>, Double>> makePathsReadersAndPriors() {
		Map<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReadersAndPriors = new HashMap<String, Tuple2<Tuple2<String, TextReader>, Double>>();

		String textPathString = textPaths;
		if (!textPaths.contains("->")) textPathString = "NoLanguageGiven->" + textPaths; // repair "invalid" input
		Map<String, String> languagePathMap = new HashMap<String, String>();
		for (String part : textPathString.split(",")) {
			String[] subparts = part.trim().split("->");
			if (subparts.length != 2) throw new IllegalArgumentException("malformed lmPath argument: comma-separated part must be of the form \"LANGUAGE->PATH\", was: " + part);
			String language = subparts[0].trim();
			String filepath = subparts[1].trim();
			languagePathMap.put(language, filepath);
		}

		Map<String, Double> languagePriorMap = new HashMap<String, Double>();
		if (languagePriors != null) {
			for (String part : languagePriors.split(",")) {
				String[] subparts = part.trim().split("->");
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
			for (String part : alternateSpellingReplacementPaths.split(",")) {
				String[] subparts = part.trim().split("->");
				if (subparts.length != 2) throw new IllegalArgumentException("malformed languagePriors argument: comma-separated part must be of the form \"LANGUAGE->PRIOR\", was: " + part);
				String language = subparts[0].trim();
				String replacementsPath = subparts[1].trim();
				if (!languagePathMap.keySet().contains(language)) throw new RuntimeException("Language '"+language+"' appears in the alternateSpellingReplacementPaths argument but not in xxx ("+languagePathMap.keySet()+")");
				languageAltSpellPathMap.put(language, replacementsPath);
			}
		}
		
		for (String language : languagePathMap.keySet()) {
			String filepath = languagePathMap.get(language);
			Double prior = languagePriorMap.get(language);
			System.out.println("For language '" + language + "', using text in " + filepath + ", prior=" + prior
					+ (languageAltSpellPathMap.keySet().contains(language) ? languageAltSpellPathMap.get(language) : ""));
			
			TextReader textReader = new BasicTextReader();
			if (languageAltSpellPathMap.keySet().contains(language)) textReader = handleReplacementRulesOption(textReader, languageAltSpellPathMap.get(language));
			if (useLongS) textReader = new ConvertLongSTextReader(textReader);
			if (!keepDiacritics) textReader = new RemoveDiacriticsTextReader(textReader);
			
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

	private Map<String, Tuple3<SingleLanguageModel, Set<String>, Double>> makeMultipleSubLMs(Map<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReadersAndPriors, Indexer<String> charIndexer) {
		Map<String, CorpusCounter> lmCounters = new HashMap<String, CorpusCounter>();
		Map<String, Set<String>> wordLists = new HashMap<String, Set<String>>();
		for (Map.Entry<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReaderAndPrior : pathsReadersAndPriors.entrySet()) {
			String language = pathsReaderAndPrior.getKey();
			String filepath = pathsReaderAndPrior.getValue()._1._1;
			TextReader textReader = pathsReaderAndPrior.getValue()._1._2;
			System.out.println(language + " text reader: " + textReader);

			CorpusCounter counter = new CorpusCounter(charN);
			//counter.countRecursive(filepath, maxLines, charIndexer, textReader);
			Tuple2<List<String>, List<String>> charsAndWords = readFileChars(filepath, textReader, lmCharCount > 0 ? lmCharCount : Long.MAX_VALUE);
			List<String> chars = charsAndWords._1;
			System.out.println("  using " + chars.size() + " characters for " + language + " read from " + filepath);
			counter.countChars(chars, charIndexer, 0);

			lmCounters.put(language, counter);
			wordLists.put(language, CollectionHelper.makeSet(charsAndWords._2));
		}
		charIndexer.lock();

		Map<String, Tuple3<SingleLanguageModel, Set<String>, Double>> lmsAndPriors = new HashMap<String, Tuple3<SingleLanguageModel, Set<String>, Double>>();
		for (Map.Entry<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReaderAndPrior : pathsReadersAndPriors.entrySet()) {
			String language = pathsReaderAndPrior.getKey();
			Double prior = pathsReaderAndPrior.getValue()._2;

			CorpusCounter counter = lmCounters.get(language);

			List<String> chars = new ArrayList<String>();
			for (int i : counter.activeCharacters)
				chars.add(charIndexer.getObject(i));
			Collections.sort(chars);
			System.out.println(language + ": " + chars);

			SingleLanguageModel lm = constructSingleLM(charIndexer, counter, counter.activeCharacters);
			lmsAndPriors.put(language, makeTuple3(lm, wordLists.get(language), prior));
		}
		return lmsAndPriors;
	}

	private Map<String, Tuple3<SingleLanguageModel, Set<String>, Double>> makeSingleSubLM(Map<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReadersAndPriors, Indexer<String> charIndexer) {
		//Map<String, Long> languageCharCounts = new HashMap<String, Long>();
		long maxCharCount = -1;
		double maxCharCountLanguagePrior = Double.NEGATIVE_INFINITY;
		double charCountLanguagePriorZ = 0.0;
		for (Map.Entry<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReaderAndPrior : pathsReadersAndPriors.entrySet()) {
			//String language = pathsReaderAndPrior.getKey();
			String filepath = pathsReaderAndPrior.getValue()._1._1;
			TextReader textReader = pathsReaderAndPrior.getValue()._1._2;
			double prior = pathsReaderAndPrior.getValue()._2;
			long charCount = readFileChars(filepath, textReader)._1.size();
			if (charCount > maxCharCount) {
				maxCharCount = charCount;
				maxCharCountLanguagePrior = prior;
			}
			charCountLanguagePriorZ += prior;
		}

		//if lmcharcountprior is not -1, then make maxcharcount that number * prior; otherwise don't.
		if (lmCharCount > 0) {
			maxCharCount = (long) (lmCharCount * maxCharCountLanguagePrior / charCountLanguagePriorZ);
			System.out.println("lmCharCount=" + lmCharCount + ", maxCharCountLanguagePrior=" + maxCharCountLanguagePrior + ", maxCharCount=" + maxCharCount);
		}

		CorpusCounter counter = new CorpusCounter(charN);
		Set<String> wordList = new HashSet<String>();
		for (Map.Entry<String, Tuple2<Tuple2<String, TextReader>, Double>> pathsReaderAndPrior : pathsReadersAndPriors.entrySet()) {
			String language = pathsReaderAndPrior.getKey();
			String filepath = pathsReaderAndPrior.getValue()._1._1;
			TextReader textReader = pathsReaderAndPrior.getValue()._1._2;
			double prior = pathsReaderAndPrior.getValue()._2;
			Tuple2<List<String>, List<String>> charsAndWords = readFileChars(filepath, textReader);
			List<String> chars = charsAndWords._1;
			wordList.addAll(charsAndWords._2);
			long charCount = chars.size();
			double multiplier = (prior / maxCharCountLanguagePrior) * maxCharCount / charCount;
			int wholeMultiplier = (int) multiplier;
			long remainderCharCount = (long) ((multiplier - wholeMultiplier) * charCount);
			for (int i = 0; i < wholeMultiplier; ++i)
				counter.countChars(chars, charIndexer, 0);

			System.out.println("For language " + language + ", multiplying " + charCount + " characters by " + multiplier + " for " + (charCount * wholeMultiplier + remainderCharCount) + " total characters");

			counter.countChars(CollectionHelper.take(chars, long2int(remainderCharCount)), charIndexer, 0);
			//counter.printStats(-1);
			System.out.println("Characters counted so far: " + counter.getTokenCount());
		}
		charIndexer.lock();

		List<String> languages = makeList(pathsReadersAndPriors.keySet());
		Collections.sort(languages);
		String language = (languages.size() == 1 ? languages.get(0) : "mixed-" + StringHelper.join(languages, "-"));

		SingleLanguageModel lm = constructSingleLM(charIndexer, counter, counter.activeCharacters);
		return Collections.singletonMap(language, makeTuple3(lm, wordList, 1.0));
	}

	private SingleLanguageModel constructSingleLM(Indexer<String> charIndexer, CorpusCounter counter, Set<Integer> activeCharacters) {
		if (!uniform)
			return new NgramLanguageModel(charIndexer, counter.getCounts(), activeCharacters, LMType.KNESER_NEY, power);
		else
			return new UniformLanguageModel(activeCharacters, charIndexer, counter.maxNgramOrder);
	}

	private Tuple2<List<String>, List<String>> readFileChars(String filepath, TextReader textReader) {
		return readFileChars(filepath, textReader, Long.MAX_VALUE);
	}

	private Tuple2<List<String>, List<String>> readFileChars(String filepath, TextReader textReader, long charsToTake) {
		List<String> allChars = new ArrayList<String>();
		List<String> allWords = new ArrayList<String>();
		outer: 
			for (File file : FileUtil.recursiveFiles(filepath)) {
				for (String line : f.readLines(file.getPath())) {
					if (allChars.size() >= charsToTake) break outer;
					List<String> chars = textReader.readCharacters(line + " ");
					allChars.addAll(chars);
					allWords.addAll(CharsetHelper.splitToEncodedWords(chars));
				}
			}
		return makeTuple2(allChars, allWords);
	}
	
	private int long2int(long l) {
		if(l > Integer.MAX_VALUE) 
			return Integer.MAX_VALUE;
		else 
			return (int)l;
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
