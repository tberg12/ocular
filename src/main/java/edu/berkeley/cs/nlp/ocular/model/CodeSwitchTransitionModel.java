package edu.berkeley.cs.nlp.ocular.model;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.setIntersection;
import indexer.Indexer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import arrays.a;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.sub.CodeSwitchGlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class CodeSwitchTransitionModel implements SparseTransitionModel {

	public class CodeSwitchTransitionState implements TransitionState {
		private final int[] context;
		public final TransitionStateType type;

		/**
		 * The current language of this state.  This may be *-1* to indicate that there is no
		 * current language state.  This will happen at, for example, the beginning of a document,
		 * where forcing a language decision before we have reached a word makes no sense.  The 
		 * null should be used to tell the system to use the language *prior* instead of the language
		 * *transition* prior.  In other words:
		 * 
		 *     p(destLang | null) = p(destLang)
		 */
		public final int langIndex;

		public final int lmCharIndex;
		public final GlyphChar glyphChar;

		public CodeSwitchTransitionState(int[] context, TransitionStateType type, int langIndex, GlyphChar glyphChar) {
			if (context == null) throw new IllegalArgumentException("context is null");
			if (glyphChar == null) throw new IllegalArgumentException("glyphChar is null");

			this.context = context;
			this.type = type;
			this.langIndex = langIndex;
			this.lmCharIndex = makeLmCharIndex(context, type);
			this.glyphChar = glyphChar;
		}

		public boolean equals(Object other) {
			if (other instanceof CodeSwitchTransitionState) {
				CodeSwitchTransitionState that = (CodeSwitchTransitionState) other;
				if (this.type != that.type || this.langIndex != that.langIndex) {
					return false;
				}
				else if (!Arrays.equals(this.context, that.context)) {
					return false;
				}
				else if (!this.glyphChar.equals(that.glyphChar)) {
					return false;
				}
				else {
					return true;
				}
			}
			else {
				return false;
			}
		}

		public int hashCode() {
			int ctxHash = Arrays.hashCode(context);
			int typeHash = this.type.ordinal();
			int langHash = this.langIndex;
			int glyphHash = this.glyphChar.hashCode();
			return 1013 * ctxHash + 1009 * typeHash + 1007 * langHash + 1017 * glyphHash;
		}

		private void addNoSubGlyphStates(List<Tuple2<TransitionState, Double>> result, int[] nextContext, TransitionStateType nextType, int nextLanguage, double transitionScore) {
			int nextLmChar = makeLmCharIndex(nextContext, nextType);
			addNoSubGlyphStates(result, nextLmChar, nextContext, nextType, nextLanguage, transitionScore);
		}
		
		private void addNoSubGlyphStates(List<Tuple2<TransitionState, Double>> result, int nextLmChar, int[] nextContext, TransitionStateType nextType, int nextLanguage, double transitionScore) {
			if (!allowGlyphSubstitution)
				addState(result, nextContext, nextType, nextLanguage, new GlyphChar(nextLmChar, false, false), transitionScore);
			else {
				GlyphType glyphType = glyphChar.toGlyphType();
				
				if (nextType == TransitionStateType.RMRGN_HPHN || nextType == TransitionStateType.RMRGN_HPHN_INIT) {
					/*
					 * This always maintains whether it is marked as a tilde-elision character 
					 * or an elided character.  This is necessary right-margin-hyphen states 
					 * in which the new state is detached from the actual previous character.
					 * Note that non-hyphen margins should just use no-sub glyph since normal
					 * (non-hyphen) margins are treated as spaces, and spaces can't be elided
					 * and can't follow tilde-elision states.
					 */
					GlyphChar nextGlyphChar = new GlyphChar(nextLmChar, glyphChar.hasElisionTilde, glyphChar.isElided);
					double glyphLogProb = calculateGlyphLogProb(nextLanguage, glyphType, lmCharIndex, nextLmChar, nextGlyphChar);
					addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
				}
				else {
					/*
					 * 1. Next state's glyph is just the rendering of the LM character
					 * 
					 * This is just a short-circuit of `addGlyphStates` in which no 
					 * substitution glyph states are permitted.  Useful for things
					 * like punctuation or spaces, where substitutions will never
					 * be allowed.
					 */
					if (glyphType != GlyphType.ELISION_TILDE) {
						// 1. Next state's glyph is just the rendering of the LM character
						GlyphChar nextGlyphChar = new GlyphChar(nextLmChar, false, false);
						double glyphLogProb = calculateGlyphLogProb(nextLanguage, glyphType, lmCharIndex, nextLmChar, nextGlyphChar);
						addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
					}
				}
			}
		}

		/**
		 * Add transition states, allowing for the possibility of substitutions or elisions.
		 * 
		 *   1. Next state's glyph is just the rendering of the LM character
		 *   2. Next state's glyph is a substitution of the LM character
		 *   3. Next state's glyph is an elision-decorated version of the LM character
		 *   4. Next state's glyph is elided
		 * 
		 */
		private void addGlyphStates(List<Tuple2<TransitionState, Double>> result, int nextLmChar, int[] nextContext, TransitionStateType nextType, int nextLanguage, double transitionScore) {
			if (!allowGlyphSubstitution)
				addState(result, nextContext, nextType, nextLanguage, new GlyphChar(nextLmChar, false, false), transitionScore);
			else {
				List<GlyphChar> potentialNextGlyphChars = new ArrayList<GlyphChar>(); 
				GlyphType glyphType = glyphChar.toGlyphType();
				if (glyphType == GlyphType.ELISION_TILDE) {
					// 4. An elision-tilde'd character must be followed by an elision
					if (canBeElided.contains(nextLmChar)) {
						potentialNextGlyphChars.add(new GlyphChar(spaceCharIndex, false, true));
					}
				}
				else {
					// 1. Next state's glyph is just the rendering of the LM character
					potentialNextGlyphChars.add(new GlyphChar(nextLmChar, false, false));
					
					// 2. Next state's glyph is a substitution of the LM character
					if (canBeReplaced.contains(nextLmChar)) {
						for (int nextGlyphCharIndex : setIntersection(codeSwitchLM.get(nextLanguage).getActiveCharacters(), validSubstitutionChars)) {
							potentialNextGlyphChars.add(new GlyphChar(nextGlyphCharIndex, false, false));
						}
					}
					
					// 3. Next state's glyph is an elision-decorated version of the LM character
					Integer tildeLmCharIndex = addTilde.get(nextLmChar);
					if (tildeLmCharIndex != null) {
						potentialNextGlyphChars.add(new GlyphChar(tildeLmCharIndex, true, false));
					}
	
					// 4. Next state's glyph is elided --- No elision can take place after a normal character 
					if (glyphType == GlyphType.ELIDED) {
						if (canBeElided.contains(nextLmChar)) {
							potentialNextGlyphChars.add(new GlyphChar(spaceCharIndex, false, true)); // TODO: commenting this out will ensure that only single-char elisions will be allowed
						}
					}
				}
					
				// Create states for all the potential next glyphs
				for (GlyphChar nextGlyphChar : potentialNextGlyphChars) {
					double glyphLogProb = calculateGlyphLogProb(nextLanguage, glyphType, lmCharIndex, nextLmChar, nextGlyphChar);
					addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
				}
			}
		}
		
		private void addTransitionsToTmpl(List<Tuple2<TransitionState, Double>> result, int[] context) {
			addTransitionsToTmpl(result, context, 0.0, false);
		}

		private void addTransitionsToTmpl(List<Tuple2<TransitionState, Double>> result, int[] context, double prevScore, boolean clearContext) {
			if (this.langIndex < 0) { // there is no current language
				for (int destLanguage = 0; destLanguage < numLanguages; ++destLanguage) { // no current language, can switch to any language
					SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
					for (int c : destLM.getActiveCharacters()) { // punctuation no problem since we have no current language
						if (c != spaceCharIndex) {
							double pDestLang = codeSwitchLM.languagePrior(destLanguage); // no language to transition from
							int[] shrunkenContext = shrinkContext(context, destLM);
							double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
							int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
							addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
						}
					}
				}
			}
			else { // there is a current language
				boolean switchAllowed = lmCharIndex == spaceCharIndex; // can switch if its (a non-space character) after a space
				if (switchAllowed) { // switch permitted
					for (int destLanguage = 0; destLanguage < numLanguages; ++destLanguage) {
						SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
						for (int c : destLM.getActiveCharacters()) {
							if (punctSet.contains(c)) {
								if (allowLanguageSwitchOnPunct) {
									double pDestLang = codeSwitchLM.languageTransitionProb(this.langIndex, destLanguage);
									int[] shrunkenContext = shrinkContext(context, destLM);
									double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
									int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
									addNoSubGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
								}
								else if (this.langIndex == destLanguage) { // switching not allowed, but this is the same language
									double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
									int[] shrunkenContext = shrinkContext(context, destLM);
									double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
									int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
									addNoSubGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
								}
							}
							else if (c != spaceCharIndex) {
								double pDestLang = codeSwitchLM.languageTransitionProb(this.langIndex, destLanguage);
								int[] shrunkenContext = shrinkContext(context, destLM);
								double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
								int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
								addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
							}
						}
					}
				}
				else { // no switching allowed
					int destLanguage = this.langIndex; // there will always be a current language here
					SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
					for (int c : destLM.getActiveCharacters()) { // punctuation no problem since we're definitely not switching anyway
						if (c != spaceCharIndex) {
							double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
							int[] shrunkenContext = shrinkContext(context, destLM);
							double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
							int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
							addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
						}
					}
				}
			}

			{ // space character: switching is never allowed
				SingleLanguageModel thisLM = codeSwitchLM.get(this.langIndex);
				// TODO: If current lmCharIndex==spaceCharIndex, sum over all languages?
				double pTransition = 0.0;
				//				if (lmCharIndex == spaceCharIndex) {
				double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
				int[] shrunkenContext = shrinkContext(context, thisLM);
				pTransition += getNgramProb(thisLM, shrunkenContext, spaceCharIndex) * pDestLang;
				//				}
				//				else {
				//					// total probability of transitioning to a space, regardless of language
				//					for (int destLanguage = 0; destLanguage < numLanguages; ++destLanguage) {
				//						SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
				//						double pDestLang = codeSwitchLM.languageTransitionPrior(this.langIndex, destLanguage);
				//						int[] shrunkenContext = shrinkContext(context, thisLM);
				//						pTransition += getNgramProb(thisLM, shrunkenContext, spaceCharIndex) * pDestLang;
				//					}
				//				}
				double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(pTransition);
				int[] nextContext = (!clearContext ? a.append(shrunkenContext, spaceCharIndex) : new int[] { spaceCharIndex });
				addNoSubGlyphStates(result, spaceCharIndex, nextContext, TransitionStateType.TMPL, this.langIndex, score);
			}
		}

		public Collection<Tuple2<TransitionState, Double>> nextLineStartStates() {
			TransitionStateType type = getType();
			int[] context = getContext();
			SingleLanguageModel thisLM = codeSwitchLM.get(this.langIndex);
			List<Tuple2<TransitionState, Double>> result = new ArrayList<Tuple2<TransitionState, Double>>();

			if (type == TransitionStateType.TMPL) {
				// transition from letter to space (left margin)
				int[] shrunkenContext = shrinkContext(context, thisLM);
				double scoreWithSpace = Math.log(getNgramProb(thisLM, shrunkenContext, spaceCharIndex));
				int[] contextWithSpace = a.append(shrunkenContext, spaceCharIndex);

				{
					double score = Math.log(LINE_MRGN_PROB) + scoreWithSpace;
					addNoSubGlyphStates(result, spaceCharIndex, contextWithSpace, TransitionStateType.LMRGN, this.langIndex, score);
				}

				addTransitionsToTmpl(result, contextWithSpace, scoreWithSpace, false);
			}
			else if (type == TransitionStateType.RMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					addNoSubGlyphStates(result, this.context, TransitionStateType.LMRGN, this.langIndex, score);
				}

				addTransitionsToTmpl(result, context);
			}
			else if (type == TransitionStateType.RMRGN_HPHN || type == TransitionStateType.RMRGN_HPHN_INIT) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					addNoSubGlyphStates(result, this.context, TransitionStateType.LMRGN_HPHN, this.langIndex, score);
				}

				if (this.langIndex >= 0) { // can't have a hyphen if there is no language, since that means there have been no characters so far
					for (int c : thisLM.getActiveCharacters()) {
						if (c != spaceCharIndex && !punctSet.contains(c)) { // can't start a line after hyphen with space or punct 
							int[] shrunkenContext = shrinkContext(context, thisLM);
							double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(thisLM, shrunkenContext, c)) /*+ Math.log(1.0)*/;
							int[] nextContext = a.append(shrunkenContext, c);
							addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, this.langIndex, score);
						}
					}
				}
			}
			else if (type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN) {
				// TODO: TAYLOR: Why do we clear the context in this case?

				{
					double score = Math.log(LINE_MRGN_PROB);
					addNoSubGlyphStates(result, new int[0], TransitionStateType.LMRGN, this.langIndex, score);
				}

				addTransitionsToTmpl(result, context, 0.0, true);
			}
			return result;
		}

		public double endLogProb() {
			return 0.0;
		}

		/**
		 * Calculate forward transitions
		 */
		public Collection<Tuple2<TransitionState, Double>> forwardTransitions() {
			TransitionStateType type = getType();
			int[] context = getContext();
			SingleLanguageModel thisLM = codeSwitchLM.get(this.langIndex);
			List<Tuple2<TransitionState, Double>> result = new ArrayList<Tuple2<TransitionState, Double>>();

			if (type == TransitionStateType.LMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					addNoSubGlyphStates(result, this.context, TransitionStateType.LMRGN, this.langIndex, score);
				}

				addTransitionsToTmpl(result, context);
			}
			else if (type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					addNoSubGlyphStates(result, this.context, TransitionStateType.LMRGN_HPHN, this.langIndex, score);
				}

				if (this.langIndex >= 0) { // can't have a hyphen if there is no language, since that means there have been no characters so far
					for (int c : thisLM.getActiveCharacters()) {
						if (c != spaceCharIndex && !punctSet.contains(c)) { // can't start a line after hyphen with space or punct 
							double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
							int[] shrunkenContext = shrinkContext(context, thisLM);
							double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(thisLM, shrunkenContext, c)) + Math.log(pDestLang);
							int[] nextContext = a.append(shrunkenContext, c);
							addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, this.langIndex, score);
						}
					}
				}
			}
			else if (type == TransitionStateType.RMRGN) {
				double score = Math.log(LINE_MRGN_PROB);
				addNoSubGlyphStates(result, this.context, TransitionStateType.RMRGN, this.langIndex, score);
			}
			else if (type == TransitionStateType.RMRGN_HPHN) {
				double score = Math.log(LINE_MRGN_PROB);
				addNoSubGlyphStates(result, this.context, TransitionStateType.RMRGN_HPHN, this.langIndex, score);
			}
			else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				double score = Math.log(LINE_MRGN_PROB);
				addNoSubGlyphStates(result, this.context, TransitionStateType.RMRGN_HPHN, this.langIndex, score);
			}
			else if (type == TransitionStateType.TMPL) {
				{
					int[] shrunkenContext = shrinkContext(context, thisLM);
					double score = Math.log(LINE_MRGN_PROB) + Math.log(1.0 - LINE_END_HYPHEN_PROB) + Math.log(getNgramProb(thisLM, shrunkenContext, spaceCharIndex));
					int[] nextContext = a.append(shrunkenContext, spaceCharIndex);
					addNoSubGlyphStates(result, spaceCharIndex, nextContext, TransitionStateType.RMRGN, this.langIndex, score);
				}

				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(LINE_END_HYPHEN_PROB);
					addNoSubGlyphStates(result, this.context, TransitionStateType.RMRGN_HPHN_INIT, this.langIndex, score);
				}

				addTransitionsToTmpl(result, context);
			}
			return result;
		}

		public int getLmCharIndex() {
			return lmCharIndex;
		}
		
		public GlyphChar getGlyphChar() {
			return glyphChar;
		}

		public int getOffset() {
			throw new Error("Method not implemented");
		}

		public int getExposure() {
			throw new Error("Method not implemented");
		}

		public int[] getContext() {
			return context;
		}

		public TransitionStateType getType() {
			return type;
		}

		public int getLanguageIndex() {
			return this.langIndex;
		}
	}

	private void addState(List<Tuple2<TransitionState, Double>> result, int[] stateContext, TransitionStateType stateType, int stateLanguage, GlyphChar glyphChar, double stateTransitionScore) {
		if (stateTransitionScore != Double.NEGATIVE_INFINITY) {
			result.add(makeTuple2((TransitionState) new CodeSwitchTransitionState(stateContext, stateType, stateLanguage, glyphChar), stateTransitionScore));
		}
	}

	public static final double LINE_MRGN_PROB = 0.5;
	public static final double LINE_END_HYPHEN_PROB = 1e-8;

	private int n;
	private int spaceCharIndex;
	private int hyphenCharIndex;
	private Set<Integer> punctSet;
	
	private Set<Integer> canBeReplaced;
	private Set<Integer> validSubstitutionChars;
	private Set<Integer> canBeElided;
	private Map<Integer, Integer> addTilde;

	private int numLanguages;
	private CodeSwitchLanguageModel codeSwitchLM;
	private CodeSwitchGlyphSubstitutionModel codeSwitchGSM;
	private boolean allowLanguageSwitchOnPunct;
	private boolean allowGlyphSubstitution;

	/**
	 * character index is the last letter of the context.
	 * 
	 * if this is the beginning of a line (context is empty or the type
	 * is a margin), then charindex is a space. if it's a right margin,
	 * then last letter is a hyphen; if there is a context then you
	 * know, context.
	 */
	public int makeLmCharIndex(int[] context, TransitionStateType type) {
		if (context.length == 0 || type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN || type == TransitionStateType.RMRGN || type == TransitionStateType.RMRGN_HPHN) {
			return spaceCharIndex;
		}
		else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
			return hyphenCharIndex;
		}
		else {
			return context[context.length - 1];
		}
	}

	/**
	 * @param languageModelsAndPriors 	<Language, <LM, PriorOfLanguage>>
	 */
	public CodeSwitchTransitionModel(CodeSwitchLanguageModel codeSwitchLM, boolean allowLanguageSwitchOnPunct, CodeSwitchGlyphSubstitutionModel codeSwitchGSM, boolean allowGlyphSubstitution) {
		this.codeSwitchLM = codeSwitchLM;
		this.codeSwitchGSM = codeSwitchGSM;
		this.allowLanguageSwitchOnPunct = allowLanguageSwitchOnPunct;
		this.allowGlyphSubstitution = allowGlyphSubstitution;

		Indexer<String> charIndexer = codeSwitchLM.getCharacterIndexer();
		this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
		this.hyphenCharIndex = charIndexer.getIndex(Charset.HYPHEN);
		this.punctSet = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.isPunctuationChar(c))
				this.punctSet.add(charIndexer.getIndex(c));
		}
		this.canBeReplaced = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.CHARS_THAT_CAN_BE_REPLACED.contains(c))
				this.canBeReplaced.add(charIndexer.getIndex(c));
		}
		this.validSubstitutionChars = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.VALID_CHAR_SUBSTITUTIONS.contains(c))
				this.validSubstitutionChars.add(charIndexer.getIndex(c));
		}
		this.canBeElided = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.CHARS_THAT_CAN_BE_ELIDED.contains(c))
				this.canBeElided.add(charIndexer.getIndex(c));
		}
		this.addTilde = new HashMap<Integer, Integer>();
		for (String c : charIndexer.getObjects()) {
			if (Charset.CHARS_THAT_CAN_BE_DECORATED_WITH_AN_ELISION_TILDE.contains(c)) {
				this.addTilde.put(charIndexer.getIndex(c), charIndexer.getIndex(Charset.TILDE_ESCAPE + c));
			}
		}

		this.n = codeSwitchLM.getMaxOrder();

		this.numLanguages = codeSwitchLM.getLanguageIndexer().size();
	}

	private void addNoSubGlyphStartState(List<Tuple2<TransitionState, Double>> result, int[] nextContext, TransitionStateType nextType, int nextLanguage, double transitionScore) {
		if (!allowGlyphSubstitution)
			addState(result, nextContext, nextType, nextLanguage, new GlyphChar(spaceCharIndex, false, false), transitionScore);
		else {
			// 1. Next state's glyph is just the rendering of the LM character
			GlyphChar nextGlyphChar = new GlyphChar(spaceCharIndex, false, false);
			double glyphLogProb = calculateGlyphLogProb(nextLanguage, GlyphType.NORMAL_CHAR, spaceCharIndex, spaceCharIndex, nextGlyphChar);
			addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
		}
	}

	/**
	 * Add transition states, allowing for the possibility of substitutions or elisions.
	 * 
	 *   1. Next state's glyph is just the rendering of the LM character
	 *   2. Next state's glyph is a substitution of the LM character
	 *   3. Next state's glyph is an elision-decorated version of the LM character
	 * X 4. Next state's glyph is elided
	 * 
	 */
	private void addGlyphStartStates(List<Tuple2<TransitionState, Double>> result, int nextLmChar, int[] nextContext, TransitionStateType nextType, int nextLanguage, double transitionScore) {
		if (!allowGlyphSubstitution)
			addState(result, nextContext, nextType, nextLanguage, new GlyphChar(nextLmChar, false, false), transitionScore);
		else {
			List<GlyphChar> potentialNextGlyphChars = new ArrayList<GlyphChar>(); 
	
			// 1. Next state's glyph is just the rendering of the LM character
			potentialNextGlyphChars.add(new GlyphChar(nextLmChar, false, false));
			
			// 2. Next state's glyph is a substitution of the LM character
			if (canBeReplaced.contains(nextLmChar)) {
				for (int nextGlyphCharIndex : setIntersection(codeSwitchLM.get(nextLanguage).getActiveCharacters(), validSubstitutionChars)) {
					potentialNextGlyphChars.add(new GlyphChar(nextGlyphCharIndex, false, false));
				}
			}
			
			// 3. Next state's glyph is an elision-decorated version of the LM character
			Integer tildeLmCharIndex = addTilde.get(nextLmChar);
			if (tildeLmCharIndex != null) {
				potentialNextGlyphChars.add(new GlyphChar(tildeLmCharIndex, true, false));
			}
	
			// Create states for all the potential next glyphs
			for (GlyphChar nextGlyphChar : potentialNextGlyphChars) {
				double glyphLogProb = calculateGlyphLogProb(nextLanguage, GlyphType.NORMAL_CHAR, spaceCharIndex, nextLmChar, nextGlyphChar);
				addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
			}
		}
	}

	/**
	 * Make a collection of states that can be the start of a line.
	 * 
	 * First possibility: L-Margin, with no context. Has probability LINE_MRGN_PROB * prior prob of the language. (1 of this) 
	 * Other possibilities: TMPL, with any individual single character c as context (~75 of these) 
	 *   - probability is: 1-LINE_MRGN_PROB * probability of c with no context * prior prob of the language.
	 *   
	 * @param d		unused
	 */
	public Collection<Tuple2<TransitionState, Double>> startStates() {
		List<Tuple2<TransitionState, Double>> result = new ArrayList<Tuple2<TransitionState, Double>>();
		/*
		 * Don't force a language choice.
		 */
		{
			double score = Math.log(LINE_MRGN_PROB) /*+ Math.log(1.0)*/;
			addNoSubGlyphStartState(result, new int[0], TransitionStateType.LMRGN, -1, score);
		}
		/*
		 * Choose among all the languages when there's an actual word (not a space).
		 */
		for (int destLanguage = 0; destLanguage < numLanguages; ++destLanguage) {
			SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
			double destLanguagePrior = codeSwitchLM.languagePrior(destLanguage);
			for (int c : destLM.getActiveCharacters()) {
				if (c != spaceCharIndex) {
					double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(destLM, new int[0], c)) + Math.log(destLanguagePrior);
					addGlyphStartStates(result, c, new int[] { c }, TransitionStateType.TMPL, destLanguage, score);
				}
			}
		}
		/*
		 * Since there's no "first" language, and we don't want to force a language 
		 * choice without an actual word, calculate the probability of starting the
		 * line with a space as the sum of the no-context space probabilities across
		 * all the languages, weighted by the language priors. 
		 */
		{
			double totalSpaceProb = 0.0;
			for (int language = 0; language < numLanguages; ++language)
				totalSpaceProb += getNgramProb(codeSwitchLM.get(language), new int[0], spaceCharIndex) * codeSwitchLM.languagePrior(language);
			double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(totalSpaceProb) /*+ Math.log(1.0)*/;
			addNoSubGlyphStartState(result, new int[] { spaceCharIndex }, TransitionStateType.TMPL, -1, score);
		}
		return result;
	}

	private double getNgramProb(SingleLanguageModel lm, int[] context, int c) {
		if (lm != null) {
			return lm.getCharNgramProb(context, c);
		}
		else {
			// No current language, so sum transition to `c` across all languages
			double totalSpaceProb = 0.0;
			for (int language = 0; language < numLanguages; ++language)
				totalSpaceProb += codeSwitchLM.get(language).getCharNgramProb(context, c) * codeSwitchLM.languagePrior(language);
			return totalSpaceProb;
		}
	}

	//	private int[] appendToContext(int[] originalContext, int c, SingleLanguageModel lm) {
	//		return shrinkContext(a.append(originalContext, c), lm);
	//	}

	private double calculateGlyphLogProb(int nextLanguage, GlyphType glyphType, int lmCharIndex, int nextLmChar, GlyphChar nextGlyphChar) {
		return codeSwitchGSM.logLanguagePrior(nextLanguage) + codeSwitchGSM.get(nextLanguage).logGlyphProb(glyphType, lmCharIndex, nextLmChar, nextGlyphChar);
	}

	private int[] shrinkContext(int[] originalContext, SingleLanguageModel lm) {
		int[] newContext = originalContext;
		while (newContext.length > n - 1)
			newContext = shortenContextForward(newContext);
		while (lm != null && !lm.containsContext(newContext)) {
			newContext = shortenContextForward(newContext);
		}
		return newContext;
	}

	private static int[] shortenContextForward(int[] context) {
		if (context.length > 0) {
			int[] result = new int[context.length - 1];
			System.arraycopy(context, 1, result, 0, result.length);
			return result;
		}
		else {
			return context;
		}
	}

}
