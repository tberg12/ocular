package edu.berkeley.cs.nlp.ocular.eval;

import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;

/**
 * @author Hannah Alpert-Abrams (halperta@gmail.com)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class LmPerplexity {

	private CodeSwitchLanguageModel lm;
	
	private final int spaceIndex;
	
	public LmPerplexity(CodeSwitchLanguageModel lm) {
		this.lm = lm;
		this.spaceIndex = lm.getCharacterIndexer().getIndex(Charset.SPACE);
	}

	public double perplexity(List<Integer> viterbiNormalizedTranscriptionCharIndices, List<Integer> viterbiNormalizedTranscriptionLangIndices) {
		double logTotalProbability = 0.0;
		for (int i=0; i<viterbiNormalizedTranscriptionCharIndices.size(); ++i) {
			int curC = viterbiNormalizedTranscriptionCharIndices.get(i);
			int curL = getLangIndex(viterbiNormalizedTranscriptionLangIndices, i);

			double langTransitionProb = getLangTransitionProb(i, curL, viterbiNormalizedTranscriptionCharIndices, viterbiNormalizedTranscriptionLangIndices);
			double ngramProb = getNgramProb(i, curC, curL, viterbiNormalizedTranscriptionCharIndices, viterbiNormalizedTranscriptionLangIndices);
			logTotalProbability += Math.log(langTransitionProb) + Math.log(ngramProb);

//			StringBuilder ctxString = new StringBuilder();
//		    for (int c: viterbiNormalizedTranscriptionCharIndices.subList(findStartPoint(i, curL, viterbiNormalizedTranscriptionLangIndices), i))
//		      ctxString.append(lm.getCharacterIndexer().getObject(c));
//		    System.out.println(String.format("P_%d(%s | %s) = %s * %s", curL, lm.getCharacterIndexer().getObject(curC), ctxString, ngramProb, langTransitionProb));
		}
		return Math.exp(-logTotalProbability / viterbiNormalizedTranscriptionCharIndices.size());
	}

	private double getNgramProb(int i, int curC, int curL, List<Integer> viterbiNormalizedTranscriptionCharIndices, List<Integer> viterbiNormalizedTranscriptionLangIndices) {
		int startPoint = findStartPoint(i, curL, viterbiNormalizedTranscriptionLangIndices);
		int[] context = CollectionHelper.intListToArray(viterbiNormalizedTranscriptionCharIndices.subList(startPoint, i));
		return lm.get(curL).getCharNgramProb(context, curC);
	}
	
	private int findStartPoint(int i, int curL, List<Integer> viterbiNormalizedTranscriptionLangIndices) {
		int startPoint = i;
		while (startPoint > 0 && getLangIndex(viterbiNormalizedTranscriptionLangIndices, startPoint-1) == curL && i-startPoint < lm.get(curL).getMaxOrder()-1) {
			--startPoint;
		}
		return startPoint;
	}

	private double getLangTransitionProb(int i, int curL, List<Integer> viterbiNormalizedTranscriptionCharIndices, List<Integer> viterbiNormalizedTranscriptionLangIndices) {
		if (i > 0) {
			int prevC = viterbiNormalizedTranscriptionCharIndices.get(i-1);
			int prevL = getLangIndex(viterbiNormalizedTranscriptionLangIndices, i-1);
			if (prevC != spaceIndex) {
				if (prevL != curL) throw new RuntimeException("Characters cannot change languages mid-word.");
				return 1.0;
			}
			else {
				return lm.languageTransitionProb(prevL, curL);
			}
		}
		else {
			return lm.languagePrior(curL);
		}
	}
	
	private int getLangIndex(List<Integer> viterbiNormalizedTranscriptionLangIndices, int i) {
		int curL = viterbiNormalizedTranscriptionLangIndices.get(i);
		if (curL < 0) {
			if (this.lm.getLanguageIndexer().size() == 1)
				curL = 0;
			else if (i > 0) 
				throw new RuntimeException("curl="+curL+", i="+i);
		}
		return curL;
	}

}
