package edu.berkeley.cs.nlp.ocular.lm;

import tberg.murphy.indexer.Indexer;

import java.util.Set;

import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class UniformLanguageModel implements SingleLanguageModel {
	private static final long serialVersionUID = 398523984923L;

	final private Set<Integer> activeCharacters;
	final private Indexer<String> charIndexer;
	final private int maxOrder;
	final private boolean[] isActive;
	final private double prob;

	public UniformLanguageModel(Set<Integer> activeCharacters, Indexer<String> charIndexer, int maxOrder) {
		this.activeCharacters = activeCharacters;
		this.charIndexer = charIndexer;
		this.maxOrder = maxOrder;

		isActive = new boolean[charIndexer.size()];
		for (int c : activeCharacters) {
			isActive[c] = true;
		}
		this.prob = 1.0 / activeCharacters.size();
	}

	public Set<Integer> getActiveCharacters() {
		return activeCharacters;
	}

	public int[] shrinkContext(int[] context) {
		return context;
	}
	
	public boolean containsContext(int[] context) {
		return true;
	}

	public double getCharNgramProb(int[] context, int c) {
		if (isActive[c])
			return prob;
		else
			return 0.0;
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	public int getMaxOrder() {
		return maxOrder;
	}

}
