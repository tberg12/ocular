package edu.berkeley.cs.nlp.ocular.lm;

import java.io.Serializable;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.lm.LanguageModel;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface SingleLanguageModel extends LanguageModel, Serializable {

	public Set<Integer> getActiveCharacters();
	public int getMaxOrder();
	public int[] shrinkContext(int[] originalContext);
	public boolean containsContext(int[] context);
	
}
