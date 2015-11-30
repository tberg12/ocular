package edu.berkeley.cs.nlp.ocular.lm;

import java.io.Serializable;

import edu.berkeley.cs.nlp.ocular.model.GlyphChar;
import edu.berkeley.cs.nlp.ocular.model.GlyphChar.GlyphType;
import indexer.Indexer;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface CodeSwitchLanguageModel extends LanguageModel, Serializable {

	public Indexer<String> getLanguageIndexer();
	
	public SingleLanguageModel get(int language);
	public double languagePrior(int language);
	public double languageTransitionPrior(int fromLanguage, int destinationLanguage);
	public double getProbKeepSameLanguage();

	public double glyphLogProb(int language, GlyphType prevGlyphChar, int prevLmChar, int lmChar, GlyphChar glyphChar);
}
