package edu.berkeley.cs.nlp.ocular.lm;

import java.io.Serializable;

import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface CodeSwitchLanguageModel extends LanguageModel, Serializable {

	public Indexer<String> getLanguageIndexer();
	
	public SingleLanguageModel get(int language);
	public double languagePrior(int language);
	public double languageTransitionProb(int fromLanguage, int destinationLanguage);
	public double getProbKeepSameLanguage();

}
