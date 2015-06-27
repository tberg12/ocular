package edu.berkeley.cs.nlp.ocular.lm;

import java.io.Serializable;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.lm.LanguageModel;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public interface SingleLanguageModel extends LanguageModel, Serializable {

	public Set<Integer> getActiveCharacters();
	public boolean containsContext(int[] context);
	
}
