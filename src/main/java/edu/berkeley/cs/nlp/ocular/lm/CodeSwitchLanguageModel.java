package edu.berkeley.cs.nlp.ocular.lm;

import java.io.Serializable;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.lm.LanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface CodeSwitchLanguageModel extends LanguageModel, Serializable {

	public Set<String> languages();

	public SingleLanguageModel get(String language);

	public Double languagePrior(String language);

	public Double languageTransitionPrior(String fromLanguage, String destinationLanguage);
	
	public Set<String> wordList(String language);

}
