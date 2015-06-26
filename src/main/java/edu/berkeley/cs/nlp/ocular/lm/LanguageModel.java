package edu.berkeley.cs.nlp.ocular.lm;

import indexer.Indexer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class LanguageModel implements Serializable {
	private static final long serialVersionUID = -4591712208091374518L;

	private Indexer<String> charIndexer;
	private CountDbBig[] countDbs;
	private int maxOrder;
	private LMType type;
	private double lmPower;
	private boolean useLongS;

	private Set<LongArrWrapper> allContextsSet;
	private List<int[]> allContexts;

	public static enum LMType { MLE, ABS_DISC, KNESER_NEY }

	public LanguageModel(Indexer<String> charIndexer, CountDbBig[] countDbs, LMType type, double lmPower, boolean useLongS) {
		this.useLongS = useLongS;
		this.charIndexer = charIndexer;
		this.countDbs = countDbs;
		this.maxOrder = countDbs.length;
		this.type = type;
		this.lmPower = lmPower;
		this.allContextsSet = new HashSet<LongArrWrapper>();
		this.allContexts = new ArrayList<int[]>();
		for (int i = 0; i < this.maxOrder - 1; i++) {
			for (long[] key : countDbs[i].getKeys()) {
				if (key != null && countDbs[i].getCount(key, CountType.HISTORY_TYPE_INDEX) > 0) {
					allContextsSet.add(new LongArrWrapper(key));
					allContexts.add(LongNgram.convertToIntArr(key));
				}
			}
		}
	}

	public static LanguageModel buildFromText(int[][] text, Indexer<String> charIndexer, int maxOrder, LMType type, double lmPower) {
		CorpusCounter counter = new CorpusCounter(maxOrder);
		counter.count(text);
		return new LanguageModel(charIndexer, counter.getCounts(), type, lmPower, false);
	}

	public static LanguageModel buildFromText(String fileName, int maxNumLines, Indexer<String> charIndexer, int maxOrder, LMType type, double lmPower, boolean useLongS) {
		CorpusCounter counter = new CorpusCounter(maxOrder);
		counter.count(fileName, maxNumLines, charIndexer, useLongS);
		return new LanguageModel(charIndexer, counter.getCounts(), type, lmPower, useLongS);
	}

	public boolean useLongS() {
		return useLongS;
	}
	
	public void checkNormalizes(int[] context) {
		double totalProb = 0;
		for (int i = 0; i < charIndexer.size(); i++) {
			totalProb += getCharNgramProb(context, i);
		}
		System.out.println("Total prob for context " + LongNgram.toString(context, charIndexer) + ": " + totalProb);
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	public int getMaxOrder() {
		return maxOrder;
	}

	public boolean containsContext(int[] context) {
		return allContextsSet.contains(new LongArrWrapper(LongNgram.convertToLong(context)));
	}

	public double getCharNgramProb(int[] context, int c) {
		// Uncomment this to renormalize the distribution after exponentiating
		//    double normalizer = 0.0;
		//    for (int i = 0; i < charIndexer.size(); i++) {
		//      normalizer += getCharNgramProbRaw(context, i);
		//    }
		//    assert normalizer > 0;
		//    return getCharNgramProbRaw(context, c)/normalizer;
		return getCharNgramProbRaw(context, c);
	}

	/**
	 * Returns an exponentiated probability, which won't necessarily
	 * sum to one
	 * @param context
	 * @param c
	 * @return
	 */
	private double getCharNgramProbRaw(int[] context, int c) {
		int[] intNgram = new int[context.length+1];
		System.arraycopy(context, 0, intNgram, 0, context.length);
		intNgram[intNgram.length-1] = c;
		NgramWrapper ngram = NgramWrapper.getNew(intNgram, 0, intNgram.length); 
		double prob = 0.0;
		switch (type) {
		case MLE: prob = new NgramCounts(ngram, countDbs).getTokenMle();
		break;
		case ABS_DISC: prob = new NgramCounts(ngram, countDbs).getAbsoluteDiscounting();
		break;
		case KNESER_NEY: prob = new NgramCounts(ngram, countDbs).getKneserNey();
		break;
		default: throw new RuntimeException("Bad type: " + type);
		}
		return Math.pow(prob, lmPower);
	}
}
