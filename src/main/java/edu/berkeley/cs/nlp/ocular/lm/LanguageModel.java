package edu.berkeley.cs.nlp.ocular.lm;

import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface LanguageModel {

	public double getCharNgramProb(int[] context, int c);
	
	public Indexer<String> getCharacterIndexer();

}
