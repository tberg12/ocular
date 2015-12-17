package edu.berkeley.cs.nlp.ocular.sub;

import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;
import indexer.Indexer;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class NoSubGlyphSubstitutionModel implements GlyphSubstitutionModel {
	private static final long serialVersionUID = 1L;

	private Indexer<String> langIndexer;
	private Indexer<String> charIndexer;

	public NoSubGlyphSubstitutionModel(Indexer<String> langIndexer, Indexer<String> charIndexer) {
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;
	}

	public Indexer<String> getLanguageIndexer() {
		return langIndexer;
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}
	
	public double glyphProb(int language, GlyphType prevGlyphType, int prevLmChar, int lmChar, GlyphChar glyphChar) {
		return (glyphChar.toGlyphType() == GlyphType.NORMAL_CHAR && lmChar == glyphChar.templateCharIndex) ? 1.0 : 0.0;
	}
	
}
