package edu.berkeley.cs.nlp.ocular.model;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.setDiff;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.setIntersection;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.intArrayToList;
import indexer.Indexer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import tuple.Pair;
import arrays.a;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class CodeSwitchTransitionModel implements SparseTransitionModel {

	public class CodeSwitchTransitionState implements TransitionState {
		private final int[] context;
		public final TransitionStateType type;

		/**
		 * The current language of this state.  This may be *null* to indicate that there is no
		 * current language state.  This will happen at, for example, the beginning of a document,
		 * where forcing a language decision before we have reached a word makes no sense.  The 
		 * null should be used to tell the system to use the language *prior* instead of the language
		 * *transition* prior.  In other words:
		 * 
		 *     p(destLang | null) = p(destLang)
		 */
		public final String language;

		public final int charIndex;

		public CodeSwitchTransitionState(int[] context, TransitionStateType type, String language) {
			if (context == null) throw new IllegalArgumentException("context is null");

			this.context = context;
			this.type = type;
			this.language = language;

			/*
			 * character index is the last letter of the context.
			 * 
			 * if this is the beginning of a line (context is empty or the type
			 * is a margin), then charindex is a space. if it's a right margin,
			 * then last letter is a hyphen; if there is a context then you
			 * know, context.
			 */
			if (context.length == 0 || type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN || type == TransitionStateType.RMRGN || type == TransitionStateType.RMRGN_HPHN) {
				this.charIndex = spaceCharIndex;
			}
			else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				this.charIndex = hyphenCharIndex;
			}
			else {
				this.charIndex = context[context.length - 1];
			}
		}

		public boolean equals(Object other) {
			if (other instanceof CodeSwitchTransitionState) {
				CodeSwitchTransitionState that = (CodeSwitchTransitionState) other;
				if (this.type != that.type || this.language != that.language) {
					return false;
				}
				else if (!Arrays.equals(this.context, that.context)) {
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
			int langHash = (this.language != null ? this.language : "_ language is null _").hashCode();
			return 1013 * ctxHash + 1009 * typeHash + 1007 * langHash;
		}

		private void addTransitionsToTmpl(List<Pair<TransitionState, Double>> result, int[] context) {
			addTransitionsToTmpl(result, context, 0.0, false);
		}

		private void addTransitionsToTmpl(List<Pair<TransitionState, Double>> result, int[] context, double prevScore, boolean clearContext) {
			if (this.language == null) { // there is no current language
				for (String destLanguage : languages) { // no current language, can switch to any language
					SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
					for (int c : destLM.getActiveCharacters()) { // punctuation no problem since we have no current language
						if (c != spaceCharIndex) {
							double pDestLang = codeSwitchLM.languagePrior(destLanguage); // no language to transition from
							int[] shrunkenContext = shrinkContext(context, destLM);
							double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
							int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
							addState(result, nextContext, TransitionStateType.TMPL, destLanguage, score);
						}
					}
				}
			}
			else { // there is a current language
				boolean switchAllowed = charIndex == spaceCharIndex; // can switch if its (a non-space character) after a space
				if (switchAllowed) { // switch permitted
					for (String destLanguage : languages) {
						SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
						// Create non-punctuation states
						for (int c : setDiff(destLM.getActiveCharacters(), punctSet)) {
							if (c != spaceCharIndex) {
								double pDestLang = codeSwitchLM.languageTransitionPrior(this.language, destLanguage);
								int[] shrunkenContext = shrinkContext(context, destLM);
								double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
								int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
								addState(result, nextContext, TransitionStateType.TMPL, destLanguage, score);
							}
						}
						// Create punctuation states
						for (int c : setIntersection(destLM.getActiveCharacters(), punctSet)) {
							if (allowLanguageSwitchOnPunct) {
								double pDestLang = codeSwitchLM.languageTransitionPrior(this.language, destLanguage);
								int[] shrunkenContext = shrinkContext(context, destLM);
								double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
								int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
								addState(result, nextContext, TransitionStateType.TMPL, destLanguage, score);
							}
							else if (this.language.equals(destLanguage)) { // switching not allowed, but this is the same language
								double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
								int[] shrunkenContext = shrinkContext(context, destLM);
								double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
								int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
								addState(result, nextContext, TransitionStateType.TMPL, destLanguage, score);
							}
						}
					}
				}
				else { // no switching allowed
					String destLanguage = this.language;
					SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
					for (int c : destLM.getActiveCharacters()) { // punctuation no problem since we're definitely not switching anyway
						if (c != spaceCharIndex) {
							double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
							int[] shrunkenContext = shrinkContext(context, destLM);
							double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, shrunkenContext, c)) + Math.log(pDestLang);
							int[] nextContext = (!clearContext ? a.append(shrunkenContext, c) : new int[] { c });
							addState(result, nextContext, TransitionStateType.TMPL, destLanguage, score);
						}
					}
				}
			}

			{ // no switching on space
				SingleLanguageModel thisLM = codeSwitchLM.get(this.language);
				// TODO: If current charIndex==spaceCharIndex, sum over all languages?
				double pTransition = 0.0;
				//				if (charIndex == spaceCharIndex) {
				double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
				int[] shrunkenContext = shrinkContext(context, thisLM);
				pTransition += getNgramProb(thisLM, shrunkenContext, spaceCharIndex) * pDestLang;
				//				}
				//				else {
				//					// total probability of transitioning to a space, regardless of language
				//					for (String destLanguage : languages) {
				//						SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
				//						double pDestLang = codeSwitchLM.languageTransitionPrior(this.language, destLanguage);
				//						int[] shrunkenContext = shrinkContext(context, thisLM);
				//						pTransition += getNgramProb(thisLM, shrunkenContext, spaceCharIndex) * pDestLang;
				//					}
				//				}
				double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(pTransition);
				int[] nextContext = (!clearContext ? a.append(shrunkenContext, spaceCharIndex) : new int[] { spaceCharIndex });
				addState(result, nextContext, TransitionStateType.TMPL, this.language, score);
			}
		}

		public Collection<Pair<TransitionState, Double>> nextLineStartStates() {
			TransitionStateType type = getType();
			int[] context = getContext();
			SingleLanguageModel thisLM = codeSwitchLM.get(this.language);
			List<Pair<TransitionState, Double>> result = new ArrayList<Pair<TransitionState, Double>>();

			if (type == TransitionStateType.TMPL) {
				// transition from letter to space (left margin)
				int[] shrunkenContext = shrinkContext(context, thisLM);
				double scoreWithSpace = Math.log(getNgramProb(thisLM, shrunkenContext, spaceCharIndex));
				int[] contextWithSpace = a.append(shrunkenContext, spaceCharIndex);

				{
					double score = Math.log(LINE_MRGN_PROB) + scoreWithSpace;
					addState(result, contextWithSpace, TransitionStateType.LMRGN, this.language, score);
				}

				addTransitionsToTmpl(result, contextWithSpace, scoreWithSpace, false);
			}
			else if (type == TransitionStateType.RMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					addState(result, this.context, TransitionStateType.LMRGN, this.language, score);
				}

				addTransitionsToTmpl(result, context);
			}
			else if (type == TransitionStateType.RMRGN_HPHN || type == TransitionStateType.RMRGN_HPHN_INIT) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					addState(result, this.context, TransitionStateType.LMRGN_HPHN, this.language, score);
				}

				if (this.language != null) { // can't have a hyphen if there is no language, since that means there have been no characters so far
					for (int c : thisLM.getActiveCharacters()) {
						if (c != spaceCharIndex && !punctSet.contains(c)) { // can't start a line after hyphen with space or punct 
							int[] shrunkenContext = shrinkContext(context, thisLM);
							double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(thisLM, shrunkenContext, c)) + Math.log(1.0);
							int[] nextContext = a.append(shrunkenContext, c);
							addState(result, nextContext, TransitionStateType.TMPL, this.language, score);
						}
					}
				}
			}
			else if (type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN) {
				// TODO: TAYLOR: Why do we clear the context in this case?

				{
					double score = Math.log(LINE_MRGN_PROB);
					addState(result, new int[0], TransitionStateType.LMRGN, this.language, score);
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
		public Collection<Pair<TransitionState, Double>> forwardTransitions() {
			TransitionStateType type = getType();
			int[] context = getContext();
			SingleLanguageModel thisLM = codeSwitchLM.get(this.language);
			List<Pair<TransitionState, Double>> result = new ArrayList<Pair<TransitionState, Double>>();

			if (type == TransitionStateType.LMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					addState(result, this.context, TransitionStateType.LMRGN, this.language, score);
				}

				addTransitionsToTmpl(result, context);
			}
			else if (type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					addState(result, this.context, TransitionStateType.LMRGN_HPHN, this.language, score);
				}

				if (this.language != null) { // can't have a hyphen if there is no language, since that means there have been no characters so far
					for (int c : thisLM.getActiveCharacters()) {
						if (c != spaceCharIndex && !punctSet.contains(c)) { // can't start a line after hyphen with space or punct 
							double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
							int[] shrunkenContext = shrinkContext(context, thisLM);
							double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(thisLM, shrunkenContext, c)) + Math.log(pDestLang);
							int[] nextContext = a.append(shrunkenContext, c);
							addState(result, nextContext, TransitionStateType.TMPL, this.language, score);
						}
					}
				}
			}
			else if (type == TransitionStateType.RMRGN) {
				double score = Math.log(LINE_MRGN_PROB);
				addState(result, this.context, TransitionStateType.RMRGN, this.language, score);
			}
			else if (type == TransitionStateType.RMRGN_HPHN) {
				double score = Math.log(LINE_MRGN_PROB);
				addState(result, this.context, TransitionStateType.RMRGN_HPHN, this.language, score);
			}
			else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				double score = Math.log(LINE_MRGN_PROB);
				addState(result, this.context, TransitionStateType.RMRGN_HPHN, this.language, score);
			}
			else if (type == TransitionStateType.TMPL) {
				{
					int[] shrunkenContext = shrinkContext(context, thisLM);
					double score = Math.log(LINE_MRGN_PROB) + Math.log(1.0 - LINE_END_HYPHEN_PROB) + Math.log(getNgramProb(thisLM, shrunkenContext, spaceCharIndex));
					int[] nextContext = a.append(shrunkenContext, spaceCharIndex);
					addState(result, nextContext, TransitionStateType.RMRGN, this.language, score);
				}

				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(LINE_END_HYPHEN_PROB);
					addState(result, this.context, TransitionStateType.RMRGN_HPHN_INIT, this.language, score);
				}

				addTransitionsToTmpl(result, context);
			}
			return result;
		}

		public int getCharIndex() {
			return charIndex;
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

		public String getLanguage() {
			return language;
		}

	}

	private void addState(List<Pair<TransitionState, Double>> result, int[] stateContext, TransitionStateType stateType, String stateLanguage, double stateTransitionScore) {
		if (stateTransitionScore != Double.NEGATIVE_INFINITY) {
			result.add(Pair.makePair((TransitionState) new CodeSwitchTransitionState(stateContext, stateType, stateLanguage), stateTransitionScore));
		}
	}

	public static final double LINE_MRGN_PROB = 0.5;
	public static final double LINE_END_HYPHEN_PROB = 1e-8;

	private int n;
	// private LanguageModel lm;
	private int spaceCharIndex;
	private int hyphenCharIndex;
	private Set<Integer> punctSet;

	private Set<String> languages;
	private CodeSwitchLanguageModel codeSwitchLM;
	private boolean allowLanguageSwitchOnPunct;
	private boolean allowDigitsInWords;

	//private Indexer<String> charIndexer;

	/**
	 * @param languageModelsAndPriors 	<Language, <LM, PriorOfLanguage>>
	 */
	public CodeSwitchTransitionModel(CodeSwitchLanguageModel codeSwitchLM, boolean allowLanguageSwitchOnPunct, boolean allowDigitsInWords) {
		this.codeSwitchLM = codeSwitchLM;
		this.allowLanguageSwitchOnPunct = allowLanguageSwitchOnPunct;
		this.allowDigitsInWords = allowDigitsInWords;

		Indexer<String> charIndexer = codeSwitchLM.getCharacterIndexer();
		this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
		this.hyphenCharIndex = charIndexer.getIndex(Charset.HYPHEN);
		this.punctSet = new HashSet<Integer>();
		for (String c : charIndexer.getObjects()) {
			if(Charset.isPunctuation(c))
				this.punctSet.add(charIndexer.getIndex(c));
		}

		this.n = codeSwitchLM.getMaxOrder();

		this.languages = codeSwitchLM.languages();
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
	public Collection<Pair<TransitionState, Double>> startStates(int d) {
		List<Pair<TransitionState, Double>> result = new ArrayList<Pair<TransitionState, Double>>();
		/*
		 * Don't force a language choice.
		 */
		{
			double score = Math.log(LINE_MRGN_PROB) + Math.log(1.0);
			addState(result, new int[0], TransitionStateType.LMRGN, null, score);
		}
		/*
		 * Choose among all the languages when there's an actual word (not a space).
		 */
		for (String destLanguage : languages) {
			SingleLanguageModel destLM = codeSwitchLM.get(destLanguage);
			double destLanguagePrior = codeSwitchLM.languagePrior(destLanguage);
			for (int c : destLM.getActiveCharacters()) {
				if (c != spaceCharIndex) {
					double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(destLM, new int[0], c)) + Math.log(destLanguagePrior);
					addState(result, new int[] { c }, TransitionStateType.TMPL, destLanguage, score);
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
			for (String language : languages)
				totalSpaceProb += getNgramProb(codeSwitchLM.get(language), new int[0], spaceCharIndex) * codeSwitchLM.languagePrior(language);
			double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(totalSpaceProb) + Math.log(1.0);
			addState(result, new int[] { spaceCharIndex }, TransitionStateType.TMPL, null, score);
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
			for (String language : languages)
				totalSpaceProb += codeSwitchLM.get(language).getCharNgramProb(context, c) * codeSwitchLM.languagePrior(language);
			return totalSpaceProb;
		}
	}

	//	private int[] appendToContext(int[] originalContext, int c, SingleLanguageModel lm) {
	//		return shrinkContext(a.append(originalContext, c), lm);
	//	}

	private int[] shrinkContext(int[] originalContext, SingleLanguageModel lm) {
		int[] newContext = originalContext;
		while (newContext.length > n - 1)
			newContext = shortenContextForward(newContext);
		while (lm != null && !lm.containsContext(newContext)) {
			if (newContext.length == 0) throw new AssertionError("shrinkContext: newContext length is zero; original context: " + intArrayToList(originalContext) + " (length=" + originalContext.length + ")");
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
