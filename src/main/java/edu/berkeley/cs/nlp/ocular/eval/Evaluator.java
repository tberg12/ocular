package edu.berkeley.cs.nlp.ocular.eval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import tberg.murphy.counter.Counter;
import tberg.murphy.counter.CounterMap;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.eval.MarkovEditDistanceComputer.EditDistanceParams;
import tberg.murphy.tuple.Pair;
import tberg.murphy.util.Iterators;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class Evaluator {

	public static class EvalSuffStats {
		private double score;
		private double docCount;

		public EvalSuffStats() {
			this.score = 0;
			this.docCount = 0;
		}
		
		public EvalSuffStats(double score, double docCount) {
			this.score = score;
			this.docCount = docCount;
		}

		public EvalSuffStats(Pair<Integer,Integer> numerDenom) {
			this.score = ((double)numerDenom.getFirst())/((double)numerDenom.getSecond());
			this.docCount = 1;
		}

		public double getScore() {
			return score;
		}

		public double getDocCount() {
			return docCount;
		}

		public void increment(EvalSuffStats suffStats) {
			double nextDocCount = getDocCount() + suffStats.getDocCount();
			double nextScore = ((getDocCount() / nextDocCount) * getScore()) + ((suffStats.getDocCount() / nextDocCount) * suffStats.getScore());
			this.score = nextScore;
			this.docCount = nextDocCount;
		}
	}

	public static String renderEval(Map<String,EvalSuffStats> evals) {
		StringBuffer buf = new StringBuffer();
		List<String> evalTypes = new ArrayList<String>(evals.keySet());
		Collections.sort(evalTypes);
		for (String evalType : evalTypes) {
			buf.append(evalType+": "+evals.get(evalType).getScore()+"\n");
		}
		return buf.toString();
	}

	public static Map<String,EvalSuffStats> getUnsegmentedEval(List<String>[] guessChars, List<String>[] goldChars, boolean charIncludesDiacritic) {
		Map<String,EvalSuffStats> evals = new HashMap<String,EvalSuffStats>();
		evals.put("CER, keep punc, allow f->s", new EvalSuffStats(getCERSuffStats(guessChars, goldChars, false, true, charIncludesDiacritic)));
		evals.put("CER, keep punc  ", new EvalSuffStats(getCERSuffStats(guessChars, goldChars, false, false, charIncludesDiacritic)));
		evals.put("CER, remove punc, allow f->s", new EvalSuffStats(getCERSuffStats(guessChars, goldChars, true, true, charIncludesDiacritic)));
		evals.put("CER, remove punc", new EvalSuffStats(getCERSuffStats(guessChars, goldChars, true, false, charIncludesDiacritic)));
		evals.put("WER, keep punc, allow f->s", new EvalSuffStats(getWERSuffStats(guessChars, goldChars, false, true)));
		evals.put("WER, keep punc  ", new EvalSuffStats(getWERSuffStats(guessChars, goldChars, false, false)));
		evals.put("WER, remove punc, allow f->s", new EvalSuffStats(getWERSuffStats(guessChars, goldChars, true, true)));
		evals.put("WER, remove punc", new EvalSuffStats(getWERSuffStats(guessChars, goldChars, true, false)));
		return evals;
	}

	public static Pair<Integer,Integer> getCERSuffStats(List<String>[] guessChars, List<String>[] goldChars, boolean removePunc, boolean allowFSConfusion, boolean charIncludesDiacritic) {
		String guessStr = fullyNormalize(guessChars, removePunc);
		String goldStr = fullyNormalize(goldChars, removePunc);
		Form guessForm = Form.charsAsGlyphs(guessStr, charIncludesDiacritic);
		Form goldForm = Form.charsAsGlyphs(goldStr, charIncludesDiacritic);
		EditDistanceParams params = EditDistanceParams.getStandardParams(guessForm, goldForm, allowFSConfusion);
		MarkovEditDistanceComputer medc = new MarkovEditDistanceComputer(params);
		AlignedFormPair alignedPair = medc.runEditDistance();
		return Pair.makePair((int)alignedPair.cost, goldForm.length());
	}

	public static Pair<Integer,Integer> getWERSuffStats(List<String>[] guessChars, List<String>[] goldChars, boolean removePunc, boolean allowFSConfusion) {
		AlignedFormPair alignedPair = getWordAlignments(guessChars, goldChars, removePunc, allowFSConfusion);
		return Pair.makePair((int)alignedPair.cost, alignedPair.trg.length());
	}

	public static String errorAnalyze(List<String>[] guessChars, List<String>[] goldChars, boolean removePunc, boolean allowFSConfusion) {
		AlignedFormPair alignedPair = getWordAlignments(guessChars, goldChars, removePunc, allowFSConfusion);
		assert alignedPair != null;
		assert alignedPair.ops != null;
		CounterMap<String,String> recallConfusions = new CounterMap<String,String>();
		Counter<String> recallErrors = new Counter<String>();
		int guessIndex = 0;
		int goldIndex = 0;
		int insertions = 0;
		int deletions = 0;
		int isolatedSubstitutions = 0;
		int nonIsolatedSubstitutions = 0;
		for (int i = 0; i < alignedPair.ops.size(); i++) {
			Operation op = alignedPair.ops.get(i);
			switch (op) {
			case EQUAL:
				guessIndex++;
				goldIndex++;
				break;
			case SUBST:
				if ((i == 0 || alignedPair.ops.get(i-1) == Operation.EQUAL) &&
						(i == alignedPair.ops.size() - 1 || alignedPair.ops.get(i+1) == Operation.EQUAL)) {
					isolatedSubstitutions++;
					recallConfusions.incrementCount(alignedPair.trg.charAt(goldIndex).toString(), alignedPair.src.charAt(guessIndex).toString(), 1.0);
				} else {
					nonIsolatedSubstitutions++;
				}
				guessIndex++;
				goldIndex++;
				break;
			case INSERT:
				insertions++;
				goldIndex++; 
				break;
			case DELETE:
				deletions++;
				guessIndex++;
				break;
			default: throw new RuntimeException("Unrecognized operation: " + op);
			}
		}
		for (String word : recallConfusions.keySet()) {
			recallErrors.incrementCount(word, recallConfusions.getCount(word));
		}
		String analysis = isolatedSubstitutions + " isolated substitutions, " + nonIsolatedSubstitutions + " non-isolated substitutions, " +
				insertions + " insertions, " + deletions + " deletions\n";
		int[] wordLengthErrorCounts = new int[10];
		int[] editDistancePerWordCounts = new int[10];
		for (Pair<String,String> wordPair : Iterators.able(recallConfusions.getPairIterator())) {
			int count = (int)recallConfusions.getCount(wordPair.getFirst(), wordPair.getSecond());
			String goldStr = wordPair.getFirst();
			String guessStr = wordPair.getSecond();
			int goldLen = Math.min(10, goldStr.length());
			wordLengthErrorCounts[goldLen-1] += 1;
			Form guessForm = Form.charsAsGlyphs(guessStr);
			Form goldForm = Form.charsAsGlyphs(goldStr);
			EditDistanceParams params = EditDistanceParams.getStandardParams(guessForm, goldForm, allowFSConfusion);
			MarkovEditDistanceComputer medc = new MarkovEditDistanceComputer(params);
			int cost = (int)medc.runEditDistance().cost;
			cost = Math.min(10, cost);
			assert cost > 0;
			editDistancePerWordCounts[cost-1] += count;
		}
		analysis += "Errors by word length (starts at 1): " + Arrays.toString(wordLengthErrorCounts) + "\n";
		analysis += "Edit distance per error (starts at 1): " + Arrays.toString(editDistancePerWordCounts) + "\n";
		analysis += "Most frequent missed words\n";
		int numPrinted = 0;
		for (String word : Iterators.able(recallErrors.asPriorityQueue())) {
			analysis += "  " + word + ": " + recallErrors.getCount(word) + "\n";
			numPrinted++;
			if (numPrinted >= 20) {
				analysis += "  ..." + recallErrors.size() + " total word types missed";
				break;
			}
		}
		return analysis;
	}

	public static AlignedFormPair getWordAlignments(List<String>[] guessChars, List<String>[] goldChars, boolean splitOutPunc, boolean allowFSConfusion) {
		String guessStr = fullyNormalize(guessChars, splitOutPunc);
		String goldStr = fullyNormalize(goldChars, splitOutPunc);
		Form guessForm = Form.wordsAsGlyphs(Arrays.asList(guessStr.split("\\s+")));
		Form goldForm = Form.wordsAsGlyphs(Arrays.asList(goldStr.split("\\s+")));
		EditDistanceParams params = EditDistanceParams.getStandardParams(guessForm, goldForm, allowFSConfusion);
		MarkovEditDistanceComputer medc = new MarkovEditDistanceComputer(params);
		AlignedFormPair alignedPair = medc.runEditDistance();
		assert alignedPair.trg.length() == goldForm.length();
		return alignedPair;
	}

	private static String fullyNormalize(List<String>[] chars, boolean splitOutPunc) {
		//    String str = convertToOneLineRemoveDashes(chars);
		String str = convertToOneLine(chars);
		//    System.out.println(str);
		//str = str.replaceAll("\\|", "s");
		if (splitOutPunc) {
			str = splitOutPunc(str);
		}
		str = normalizeWhitespace(str);
		//    System.out.println("Normalized: <begin>" + str + "<end>");
		//    System.out.println(str);
		return str;
	}

	@SuppressWarnings("unused")
	private static String convertToOneLineRemoveDashes(List<String>[] chars) {
		String str = "";
		for (List<String> line : chars) {
			String lineString = "";
			for (int i = 0; i < line.size(); i++) {
				lineString += line.get(i);
			}
			lineString = lineString.trim();
			if (str.endsWith("-")) {
				str = str.substring(0,str.length()-1) + lineString;
			} else {
				str = str.substring(0,str.length()) + " " + lineString;
			}
		}
		return str;
	}

	private static String convertToOneLine(List<String>[] chars) {
		String str = "";
		for (List<String> line : chars) {
			for (int i = 0; i < line.size(); i++) {
				str += line.get(i);
			}
			str += " ";
		}
		return str;
	}

	private static String normalizeWhitespace(String str) {
		return str.trim().replaceAll("\\s+", " ");
	}

	private static String splitOutPunc(String str) {
		StringBuffer buf = new StringBuffer();
		for (String c: Charset.readNormalizeCharacters(str)) {
			if (!Charset.isPunctuationChar(c)) buf.append(c);
		}
		return normalizeWhitespace(buf.toString());
	}

	public static void main(String[] args) {
		String guess = "this is a longer, more nuanced test of the system";
		String gold = "tis is a logner, more nunced test of the sstem";
		System.out.println(renderEval(getUnsegmentedEval(convertToLines(guess), convertToLines(gold), true)));

		String guess2 = "deletion deletion this is a longer, more nuanced test of the system";
		String gold2 = "tis is a logner, more nunced test of the sstem insertion insertion";
		System.out.println(renderEval(getUnsegmentedEval(convertToLines(guess2), convertToLines(gold2), true)));

		String guess3 = "this is a longer, more nuanced test of the system deletion deletion";
		String gold3 = "insertion insertion tis is a logner, more nunced test of the sstem";
		System.out.println(renderEval(getUnsegmentedEval(convertToLines(guess3), convertToLines(gold3), true)));

		String guess4 = "this is \n a longer, more\n nuan-\nced  \n   test of the system deletion deletion";
		String gold4 = "this is a lon- \n ger, more nuanced test of the system deletion deletion";
		System.out.println(renderEval(getUnsegmentedEval(convertToLines(guess4), convertToLines(gold4), true)));

		String guess5 = "this is a longer, more nuanced t\\'est of the system";
		String gold5 = "tis is a logner, more nunced t\\'est of the sstem";
		System.out.println(renderEval(getUnsegmentedEval(convertToLines(guess5), convertToLines(gold5), true)));
	}

	private static List<String>[] convertToLines(String rawStr) {
		String[] lines = rawStr.split("\n");
		@SuppressWarnings("unchecked")
		List<String>[] charsPerLine = new List[lines.length];
		for (int i = 0; i < lines.length; i++) {
			charsPerLine[i] = Arrays.asList(split(lines[i]));
		}
		return charsPerLine;
	}

	public static String[] split(String str) {
		String[] result = new String[str.length()];
		for (int i=0; i<result.length; ++i) {
			result[i] = str.substring(i, i+1);
		}
		return result;
	}


}
