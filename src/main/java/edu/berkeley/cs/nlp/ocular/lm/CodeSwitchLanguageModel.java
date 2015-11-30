package edu.berkeley.cs.nlp.ocular.lm;

import java.io.Serializable;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.model.GlyphChar;
import edu.berkeley.cs.nlp.ocular.model.GlyphChar.GlyphType;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface CodeSwitchLanguageModel extends LanguageModel, Serializable {

	public Set<String> languages();

	public SingleLanguageModel get(String language);

	public Double languagePrior(String language);

	public Double languageTransitionPrior(String fromLanguage, String destinationLanguage);
	
	public double glyphLogProb(String language, GlyphType prevGlyphChar, int prevLmChar, int lmChar, GlyphChar glyphChar);

	public double getProbKeepSameLanguage();
	
}
