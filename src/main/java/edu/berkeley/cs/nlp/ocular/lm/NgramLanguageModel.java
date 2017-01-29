package edu.berkeley.cs.nlp.ocular.lm;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.textreader.CharIndexer;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.data.textreader.TextReader;
import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;
import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import tberg.murphy.indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class NgramLanguageModel implements SingleLanguageModel {
	private static final long serialVersionUID = 873286328149782L;

	private Indexer<String> charIndexer;
	private CountDbBig[] countDbs;
	private int maxOrder;
	private LMType type;
	private double lmPower;

	private Set<LongArrWrapper> allContextsSet;
	private List<int[]> allContexts;

	public static enum LMType { MLE, ABS_DISC, KNESER_NEY }
	
	private Set<Integer> activeCharacters;
	public Set<Integer> getActiveCharacters() { return activeCharacters; }

	public NgramLanguageModel(Indexer<String> charIndexer, CountDbBig[] countDbs, Set<Integer> activeCharacters, LMType type, double lmPower) {
		this.charIndexer = charIndexer;
		this.countDbs = countDbs;
		this.maxOrder = countDbs.length;
		if (maxOrder <= 0) throw new RuntimeException("maxOrder must be greater than zero.");
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
		
		if (activeCharacters == null) throw new RuntimeException("activeCharacters is null!"); 
		this.activeCharacters = activeCharacters;
	}

	public static NgramLanguageModel buildFromText(String fileName, int maxNumLines, int maxOrder, LMType type, double lmPower, TextReader textReader) {
		return buildFromText(CollectionHelper.makeList(fileName), maxNumLines, maxOrder, type, lmPower, textReader);
	}

	public static NgramLanguageModel buildFromText(List<String> fileNames, int maxNumLines, int maxOrder, LMType type, double lmPower, TextReader textReader) {
		CorpusCounter counter = new CorpusCounter(maxOrder);
		Set<Integer> activeCharacters = counter.getActiveCharacters();
		Indexer<String> charIndexer = new CharIndexer();
		for (String fileName : fileNames) {
			counter.countRecursive(fileName, maxNumLines, charIndexer, textReader);
		}
		activeCharacters.add(charIndexer.getIndex(Charset.SPACE));
		charIndexer.lock();
		counter.printStats(-1);
		return new NgramLanguageModel(charIndexer, counter.getCounts(), activeCharacters, type, lmPower);
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

	public double getLmPower() {
		return lmPower;
	}

	public int[] shrinkContext(int[] originalContext) {
		int[] newContext = originalContext;
		if (newContext.length > maxOrder - 1) {
			newContext = ArrayHelper.takeRight(newContext, maxOrder - 1);
		}
		while (!containsContext(newContext) && newContext.length > 0) {
			newContext = ArrayHelper.takeRight(newContext, newContext.length - 1);
		}
		return newContext;
	}
	
	public boolean containsContext(int[] context) {
		if (context.length == 0)
			return true;
		else
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
