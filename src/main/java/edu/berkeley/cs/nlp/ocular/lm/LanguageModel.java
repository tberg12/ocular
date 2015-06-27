package edu.berkeley.cs.nlp.ocular.lm;

import indexer.Indexer;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface LanguageModel {

	public double getCharNgramProb(int[] context, int c);
	
	public Indexer<String> getCharacterIndexer();
	public int getMaxOrder();

}
