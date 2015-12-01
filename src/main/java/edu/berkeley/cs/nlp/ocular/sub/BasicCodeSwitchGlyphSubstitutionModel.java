package edu.berkeley.cs.nlp.ocular.sub;

import java.io.Serializable;
import java.util.List;

import indexer.Indexer;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class BasicCodeSwitchGlyphSubstitutionModel implements CodeSwitchGlyphSubstitutionModel, Serializable {
	private static final long serialVersionUID = 1L;

	private Indexer<String> langIndexer;
	private List<Double> logdistLanguagePrior;
	private List<SingleGlyphSubstitutionModel> subModels;

	public BasicCodeSwitchGlyphSubstitutionModel(Indexer<String> langIndexer, List<Double> logdistLanguagePrior, List<SingleGlyphSubstitutionModel> subModels) {
		this.langIndexer = langIndexer;
		this.logdistLanguagePrior = logdistLanguagePrior;
		this.subModels = subModels;
	}

	public Indexer<String> getLanguageIndexer() {
		return langIndexer;
	}

	public SingleGlyphSubstitutionModel get(int language) {
		if (language == -1) // no language selected
			return null;
		else
			return subModels.get(language);
	}

	public double logLanguagePrior(int language) {
		return logdistLanguagePrior.get(language);
	}

}
