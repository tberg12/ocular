package edu.berkeley.cs.nlp.ocular.model.transition;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeAddTildeMap;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeElidedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeCanBeReplacedSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeDiacriticDisregardMap;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makePunctSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeValidDoublableSet;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.makeValidSubstitutionCharsSet;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeSet;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import tberg.murphy.arrays.a;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
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
				addState(result, nextContext, nextType, nextLanguage, new GlyphChar(nextLmChar, GlyphType.NORMAL_CHAR), transitionScore);
			else {
				GlyphType glyphType = glyphChar.glyphType;
				
				if (nextType == TransitionStateType.RMRGN_HPHN_INIT || nextType == TransitionStateType.RMRGN_HPHN|| nextType == TransitionStateType.LMRGN_HPHN) {
					/*
					 * This always maintains whether it is marked as a tilde-elision character 
					 * or an elided character.  This is necessary right-margin-hyphen states 
					 * in which the new state is detached from the actual previous character.
					 * Note that non-hyphen margins should just use no-sub glyph since normal
					 * (non-hyphen) margins are treated as spaces, and spaces can't be elided
					 * and can't follow tilde-elision states.
					 */
					{
						GlyphChar nextGlyphChar = new GlyphChar(nextLmChar, glyphChar.glyphType);
						double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphChar);
						addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
					}
					
					if (nextType == TransitionStateType.RMRGN_HPHN_INIT) {
						/*
						 * Allow for the elision of Ouptut a space 
						 */
						GlyphChar nextGlyphChar = new GlyphChar(spaceCharIndex, glyphChar.glyphType);
						double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphChar);
						addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
					}
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
					if (glyphType != GlyphType.ELISION_TILDE) { // normal state can't follow an elision-marking tilde
						// 1. Next state's glyph is just the rendering of the LM character
						GlyphChar nextGlyphChar = new GlyphChar(nextLmChar, GlyphType.NORMAL_CHAR);
						double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphChar);
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
		 *   4. Next state's glyph is an elision after a tilde-decorated character
		 *   5. Next state's glyph is the LM char, stripped of its accents
		 *   6. Next state's glyph is an elision after a space
		 *   7. Next state's glyph is a doubled version of the LM character
		 * 
		 */
		private void addGlyphStates(List<Tuple2<TransitionState, Double>> result, int nextLmChar, int[] nextContext, TransitionStateType nextType, int nextLanguage, double transitionScore) {
			if (!allowGlyphSubstitution)
				addState(result, nextContext, nextType, nextLanguage, new GlyphChar(nextLmChar, GlyphType.NORMAL_CHAR), transitionScore);
			else {
				Set<GlyphChar> potentialNextGlyphChars = new HashSet<GlyphChar>(); 
				GlyphType glyphType = glyphChar.glyphType;
				if (glyphType == GlyphType.DOUBLED) {
					// Deterministically duplicate the glyph (but no longer marked as "doubled")
					//potentialNextGlyphChars.add(new GlyphChar(glyphChar.templateCharIndex, GlyphType.NORMAL_CHAR));
					throw new RuntimeException("This should have been handled elsewhere so that we don't re-include ngram LM scores");
				}
				else if (glyphType == GlyphType.ELISION_TILDE) {
					// 4. An elision-tilde'd character must be followed by a tilde-elision
					if (canBeElided.contains(nextLmChar)) {
						potentialNextGlyphChars.add(new GlyphChar(spaceCharIndex, GlyphType.TILDE_ELIDED));
					}
				}
				else {
					// 1. Next state's glyph is just the rendering of the LM character
					potentialNextGlyphChars.add(new GlyphChar(nextLmChar, GlyphType.NORMAL_CHAR));

					// 2. Next state's glyph is a substitution of the LM character
					if (canBeReplaced.contains(nextLmChar)) {
						for (int nextGlyphCharIndex : lm.get(nextLanguage).getActiveCharacters()) {
							if (validSubstitutionChars.contains(nextGlyphCharIndex)) {
								potentialNextGlyphChars.add(new GlyphChar(nextGlyphCharIndex, GlyphType.NORMAL_CHAR));
							}
						}
					}
					if (nextLmChar == sCharIndex)
						potentialNextGlyphChars.add(new GlyphChar(longsCharIndex, GlyphType.NORMAL_CHAR));
					
					// 3. Next state's glyph is an elision-decorated version of the LM character
					Integer tildeDecorated = addTilde.get(nextLmChar);
					if (tildeDecorated != null) {
						potentialNextGlyphChars.add(new GlyphChar(tildeDecorated, GlyphType.ELISION_TILDE));
					}
	
					// 4. Next state's glyph is elided --- No elision can take place after a normal character 
					if (glyphType == GlyphType.TILDE_ELIDED) {
						if (canBeElided.contains(nextLmChar)) {
							potentialNextGlyphChars.add(new GlyphChar(spaceCharIndex, GlyphType.TILDE_ELIDED));
						}
					}
					
					// 5. Next state's glyph is the LM char, stripped of its accents
					Integer baseChar = diacriticDisregardMap.get(nextLmChar);
					if (baseChar != null) {
						potentialNextGlyphChars.add(new GlyphChar(baseChar, GlyphType.NORMAL_CHAR));
					}
					
					// 6. Next state's glyph is an elision after a space
					if (!elideAnything) {
					if (glyphType != GlyphType.FIRST_ELIDED) { // TODO: Comment this out if we want to allow multiple characters to be elided from the front of a word
						if (lmCharIndex == spaceCharIndex) {
							if (type != TransitionStateType.LMRGN_HPHN && type != TransitionStateType.RMRGN_HPHN_INIT && type != TransitionStateType.RMRGN_HPHN) { // only allowed at the start of a word, not in the middle of a hyphenated word
								if (nextType == TransitionStateType.TMPL) {
									if (canBeElided.contains(nextLmChar)) {
										potentialNextGlyphChars.add(new GlyphChar(spaceCharIndex, GlyphType.FIRST_ELIDED));
									}
								}
							}
						}
					}
					}
					
					// 7. Next state's glyph is a doubled version of the LM character
					if (validDoublableSet.contains(nextLmChar)) {
						potentialNextGlyphChars.add(new GlyphChar(nextLmChar, GlyphType.DOUBLED));
						if (nextLmChar == sCharIndex)
							potentialNextGlyphChars.add(new GlyphChar(longsCharIndex, GlyphType.DOUBLED));
					}
					
					// 8. Elide the character
					if (elideAnything) {
						if (nextType == TransitionStateType.TMPL) {
							if (canBeElided.contains(nextLmChar)) {
								potentialNextGlyphChars.add(new GlyphChar(spaceCharIndex, GlyphType.ELIDED));
							}
						}
					}

				}
				
				// Create states for all the potential next glyphs
				for (GlyphChar nextGlyphChar : potentialNextGlyphChars) {
					double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphChar);
					addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
				}
			}
		}
		
		private void addTransitionsToTmpl(List<Tuple2<TransitionState, Double>> result, int[] context) {
			addTransitionsToTmpl(result, context, 0.0, false);
		}

		private void addTransitionsToTmpl(List<Tuple2<TransitionState, Double>> result, int[] context, double prevScore, boolean clearContext) {
			if (glyphChar.glyphType == GlyphType.DOUBLED) {
				// Duplicate the state: same context, language, lmChar, ...; but Doubled=>Normal
				TransitionStateType nextType = TransitionStateType.TMPL;
				int nextLanguage = langIndex;
				int nextLmChar = lmCharIndex;
				//SingleLanguageModel destLM = lm.get(nextLanguage);
				//double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
				double score = prevScore; //+ Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(destLM, context, nextLmChar)) + Math.log(pDestLang); // TODO: Is it necessary to have some sort of LM probability factored in?
				if (nextLmChar == sCharIndex) { // a doubled 's' may have long-s chars
					GlyphChar nextGlyphCharS = new GlyphChar(sCharIndex, GlyphType.NORMAL_CHAR);
					double glyphLogProbS = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphCharS);
					addState(result, context, nextType, nextLanguage, nextGlyphCharS, score + glyphLogProbS);

					GlyphChar nextGlyphCharLongs = new GlyphChar(longsCharIndex, GlyphType.NORMAL_CHAR);
					double glyphLogProbLongs = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphCharLongs);
					addState(result, context, nextType, nextLanguage, nextGlyphCharLongs, score + glyphLogProbLongs);
				}
				else {
					GlyphChar nextGlyphChar = new GlyphChar(lmCharIndex, GlyphType.NORMAL_CHAR);
					double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphChar);
					addState(result, context, nextType, nextLanguage, nextGlyphChar, score + glyphLogProb);
				}
			}
			else {
				if (this.langIndex < 0) { // there is no current language
					for (int destLanguage = 0; destLanguage < numLanguages; ++destLanguage) { // no current language, can switch to any language
						SingleLanguageModel destLM = lm.get(destLanguage);
						for (int c : destLM.getActiveCharacters()) { // punctuation no problem since we have no current language
							if (c != spaceCharIndex) {
								double pDestLang = lm.languagePrior(destLanguage); // no language to transition from
								double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, context, c)) + Math.log(pDestLang);
								int[] nextContext = (!clearContext ? a.append((destLM!=null ? shrinkContext(context, destLM) : context), c) : new int[] { c });
								addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
							}
						}
					}
				}
				else { // there is a current language
					boolean switchAllowed = lmCharIndex == spaceCharIndex; // can switch if its (a non-space character) after a space
					if (switchAllowed) { // switch permitted
						for (int destLanguage = 0; destLanguage < numLanguages; ++destLanguage) {
							SingleLanguageModel destLM = lm.get(destLanguage);
							for (int c : destLM.getActiveCharacters()) {
								if (punctSet.contains(c)) {
									if (allowLanguageSwitchOnPunct) {
										double pDestLang = lm.languageTransitionProb(this.langIndex, destLanguage);
										double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, context, c)) + Math.log(pDestLang);
										int[] nextContext = (!clearContext ? a.append((destLM!=null ? shrinkContext(context, destLM) : context), c) : new int[] { c });
										addNoSubGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
									}
									else if (this.langIndex == destLanguage) { // switching not allowed, but this is the same language
										double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
										double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, context, c)) + Math.log(pDestLang);
										int[] nextContext = (!clearContext ? a.append((destLM!=null ? shrinkContext(context, destLM) : context), c) : new int[] { c });
										addNoSubGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
									}
								}
								else if (c != spaceCharIndex) {
									double pDestLang = lm.languageTransitionProb(this.langIndex, destLanguage);
									double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, context, c)) + Math.log(pDestLang);
									int[] nextContext = (!clearContext ? a.append((destLM!=null ? shrinkContext(context, destLM) : context), c) : new int[] { c });
									addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
								}
							}
						}
					}
					else { // no switching allowed
						int destLanguage = this.langIndex; // there will always be a current language here
						SingleLanguageModel destLM = lm.get(destLanguage);
						for (int c : destLM.getActiveCharacters()) { // punctuation no problem since we're definitely not switching anyway
							if (c != spaceCharIndex) {
								double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
								double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(getNgramProb(destLM, context, c)) + Math.log(pDestLang);
								int[] nextContext = (!clearContext ? a.append((destLM!=null ? shrinkContext(context, destLM) : context), c) : new int[] { c });
								addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, destLanguage, score);
							}
						}
					}
				}
	
				{ // space character: switching is never allowed
					SingleLanguageModel thisLM = lm.get(this.langIndex);
					// TODO: If current lmCharIndex==spaceCharIndex, sum over all languages?
					double pTransition = 0.0;
					//				if (lmCharIndex == spaceCharIndex) {
					double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
					pTransition += getNgramProb(thisLM, context, spaceCharIndex) * pDestLang;
					//				}
					//				else {
					//					// total probability of transitioning to a space, regardless of language
					//					for (int destLanguage = 0; destLanguage < numLanguages; ++destLanguage) {
					//						SingleLanguageModel destLM = lm.get(destLanguage);
					//						double pDestLang = lm.languageTransitionPrior(this.langIndex, destLanguage);
					//						int[] shrunkenContext = shrinkContext(context, thisLM);
					//						pTransition += getNgramProb(thisLM, context, spaceCharIndex) * pDestLang;
					//					}
					//				}
					double score = Math.log(1.0 - LINE_MRGN_PROB) + prevScore + Math.log(pTransition);
					int[] nextContext = (!clearContext ? a.append((thisLM!=null ? shrinkContext(context, thisLM) : context), spaceCharIndex) : new int[] { spaceCharIndex });
					addNoSubGlyphStates(result, spaceCharIndex, nextContext, TransitionStateType.TMPL, this.langIndex, score);
				}
			}
		}

		public Collection<Tuple2<TransitionState, Double>> nextLineStartStates() {
			SingleLanguageModel thisLM = lm.get(this.langIndex);
			List<Tuple2<TransitionState, Double>> result = new ArrayList<Tuple2<TransitionState, Double>>();

			if (type == TransitionStateType.TMPL) {
				// transition from letter to space (left margin)
				double scoreWithSpace = Math.log(getNgramProb(thisLM, context, spaceCharIndex));
				int[] contextWithSpace = a.append((thisLM!=null ? shrinkContext(context, thisLM) : context), spaceCharIndex);

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
					if (glyphChar.glyphType == GlyphType.DOUBLED) {
						// Duplicate the state: same context, language, lmChar, ...; but Doubled=>Normal
						TransitionStateType nextType = TransitionStateType.TMPL;
						int nextLanguage = langIndex;
						int nextLmChar = lmCharIndex;
						double score = Math.log(1.0); //+ Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(thisLM, context, nextLmChar)) + Math.log(1.0); // TODO: Is it necessary to have some sort of LM probability factored in?
						if (nextLmChar == sCharIndex) { // a doubled 's' may have long-s chars
							GlyphChar nextGlyphCharS = new GlyphChar(sCharIndex, GlyphType.NORMAL_CHAR);
							double glyphLogProbS = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphCharS);
							addState(result, context, nextType, nextLanguage, nextGlyphCharS, score + glyphLogProbS);

							GlyphChar nextGlyphCharLongs = new GlyphChar(longsCharIndex, GlyphType.NORMAL_CHAR);
							double glyphLogProbLongs = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphCharLongs);
							addState(result, context, nextType, nextLanguage, nextGlyphCharLongs, score + glyphLogProbLongs);
						}
						else {
							GlyphChar nextGlyphChar = new GlyphChar(lmCharIndex, GlyphType.NORMAL_CHAR);
							double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphChar);
							addState(result, context, nextType, nextLanguage, nextGlyphChar, score + glyphLogProb);
						}
					}
					else {
						for (int c : thisLM.getActiveCharacters()) {
							if (c != spaceCharIndex && !punctSet.contains(c)) { // can't start a line after hyphen with space or punct 
								double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(thisLM, context, c)) /*+ Math.log(1.0)*/;
								int[] nextContext = a.append((thisLM!=null ? shrinkContext(context, thisLM) : context), c);
								addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, this.langIndex, score);
							}
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
			if (glyphChar.glyphType == GlyphType.DOUBLED || glyphChar.glyphType == GlyphType.ELISION_TILDE) // can't end on an incomplete "double glyph"
				return Double.NEGATIVE_INFINITY;
			else
				return 0.0;
		}

		/**
		 * Calculate forward transitions
		 */
		public Collection<Tuple2<TransitionState, Double>> forwardTransitions() {
			SingleLanguageModel thisLM = lm.get(this.langIndex);
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
					if (glyphChar.glyphType == GlyphType.DOUBLED) {
						// Duplicate the state: same context, language, lmChar, ...; but Doubled=>Normal
						TransitionStateType nextType = TransitionStateType.TMPL;
						int nextLanguage = langIndex;
						int nextLmChar = lmCharIndex;
						//double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
						double score = Math.log(1.0); //+ Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(thisLM, context, nextLmChar)) + Math.log(pDestLang); // TODO: Is it necessary to have some sort of LM probability factored in?
						if (nextLmChar == sCharIndex) { // a doubled 's' may have long-s chars
							GlyphChar nextGlyphCharS = new GlyphChar(sCharIndex, GlyphType.NORMAL_CHAR);
							double glyphLogProbS = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphCharS);
							addState(result, context, nextType, nextLanguage, nextGlyphCharS, score + glyphLogProbS);

							GlyphChar nextGlyphCharLongs = new GlyphChar(longsCharIndex, GlyphType.NORMAL_CHAR);
							double glyphLogProbLongs = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphCharLongs);
							addState(result, context, nextType, nextLanguage, nextGlyphCharLongs, score + glyphLogProbLongs);
						}
						else {
							GlyphChar nextGlyphChar = new GlyphChar(lmCharIndex, GlyphType.NORMAL_CHAR);
							double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphChar);
							addState(result, context, nextType, nextLanguage, nextGlyphChar, score + glyphLogProb);
						}
					}
					else {
						for (int c : thisLM.getActiveCharacters()) {
							if (c != spaceCharIndex && !punctSet.contains(c)) { // can't start a line after hyphen with space or punct 
								double pDestLang = 1.0; // since there's only one language for this character, don't divide its mass across languages
								double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(getNgramProb(thisLM, context, c)) + Math.log(pDestLang);
								int[] nextContext = a.append((thisLM!=null ? shrinkContext(context, thisLM) : context), c);
								addGlyphStates(result, c, nextContext, TransitionStateType.TMPL, this.langIndex, score);
							}
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
					double score = Math.log(LINE_MRGN_PROB) + Math.log(1.0 - LINE_END_HYPHEN_PROB) + Math.log(getNgramProb(thisLM, context, spaceCharIndex));
					int[] nextContext = a.append((thisLM!=null ? shrinkContext(context, thisLM) : context), spaceCharIndex);
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

		public TransitionStateType getType() {
			return type;
		}

		public int getLanguageIndex() {
			return this.langIndex;
		}
		
		public String toString() {
			StringBuilder contextSB = new StringBuilder("[");
			for (int c : context)
				contextSB.append(charIndexer.getObject(c));
				//.append(", ");
			//if (context.length > 0) contextSB.delete(contextSB.length()-2, contextSB.length());
			contextSB.append("]");
			return "CodeSwitchTransitionState("+(langIndex>=0 ? langIndexer.getObject(langIndex) : "No Language")+", "+charIndexer.getObject(lmCharIndex)+", "+type+", "+contextSB+", "+glyphChar.toString(charIndexer)+")";
		}
	}

	private void addState(List<Tuple2<TransitionState, Double>> result, int[] stateContext, TransitionStateType stateType, int stateLanguage, GlyphChar glyphChar, double stateTransitionScore) {
		if (stateTransitionScore != Double.NEGATIVE_INFINITY) {
			result.add(Tuple2((TransitionState) new CodeSwitchTransitionState(stateContext, stateType, stateLanguage, glyphChar), stateTransitionScore));
		}
	}

	public static final double LINE_MRGN_PROB = 0.5;
	public static final double LINE_END_HYPHEN_PROB = 1e-8;

	private Indexer<String> charIndexer;
	private Indexer<String> langIndexer;
	private int spaceCharIndex;
	private int hyphenCharIndex;
	private int sCharIndex;
	private int longsCharIndex;
	private Set<Integer> punctSet;
	
	private Set<Integer> canBeReplaced;
	private Set<Integer> validSubstitutionChars;
	private Set<Integer> validDoublableSet;
	private Set<Integer> canBeElided;
	private Map<Integer, Integer> addTilde;
	private Map<Integer,Integer> diacriticDisregardMap;

	private int numLanguages;
	private CodeSwitchLanguageModel lm;
	private GlyphSubstitutionModel gsm;
	private boolean allowLanguageSwitchOnPunct;
	private boolean allowGlyphSubstitution;
	private double noCharSubPrior;
	private boolean elideAnything;

	private Set<TransitionStateType> alwaysSpaceTransitionTypes;
	
	/**
	 * character index is the last letter of the context.
	 * 
	 * if this is the beginning of a line (context is empty or the type
	 * is a margin), then charindex is a space. if it's a right margin,
	 * then last letter is a hyphen; if there is a context then you
	 * know, context.
	 */
	private int makeLmCharIndex(int[] context, TransitionStateType type) {
		if (context.length == 0 || this.alwaysSpaceTransitionTypes.contains(type)) {
			return spaceCharIndex;
		}
		else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
			return hyphenCharIndex;
		}
		else {
			return context[context.length - 1];
		}
	}

	public CodeSwitchTransitionModel(CodeSwitchLanguageModel lm, boolean allowLanguageSwitchOnPunct, GlyphSubstitutionModel gsm, boolean allowGlyphSubstitution, double noCharSubPrior, boolean elideAnything) {
		this.lm = lm;
		this.gsm = gsm;
		this.allowLanguageSwitchOnPunct = allowLanguageSwitchOnPunct;
		this.allowGlyphSubstitution = allowGlyphSubstitution;
		this.noCharSubPrior = noCharSubPrior;
		this.elideAnything = elideAnything;

		this.charIndexer = lm.getCharacterIndexer();
		this.langIndexer = lm.getLanguageIndexer();
		this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
		this.hyphenCharIndex = charIndexer.getIndex(Charset.HYPHEN);
		this.sCharIndex = charIndexer.contains("s") ? charIndexer.getIndex("s") : -1;
		this.longsCharIndex = charIndexer.getIndex(Charset.LONG_S);
		this.punctSet = makePunctSet(charIndexer);
		this.canBeReplaced = makeCanBeReplacedSet(charIndexer);
		this.validSubstitutionChars = makeValidSubstitutionCharsSet(charIndexer);
		this.validDoublableSet = makeValidDoublableSet(charIndexer);
		this.canBeElided = makeCanBeElidedSet(charIndexer);
		this.addTilde = makeAddTildeMap(charIndexer);
		this.diacriticDisregardMap = makeDiacriticDisregardMap(charIndexer);

		this.numLanguages = lm.getLanguageIndexer().size();
		this.alwaysSpaceTransitionTypes = makeSet(TransitionStateType.LMRGN, TransitionStateType.LMRGN_HPHN, TransitionStateType.RMRGN, TransitionStateType.RMRGN_HPHN);
	}

	private void addNoSubGlyphStartState(List<Tuple2<TransitionState, Double>> result, int[] nextContext, TransitionStateType nextType, int nextLanguage, double transitionScore) {
		if (!allowGlyphSubstitution)
			addState(result, nextContext, nextType, nextLanguage, new GlyphChar(spaceCharIndex, GlyphType.NORMAL_CHAR), transitionScore);
		else {
			// 1. Next state's glyph is just the rendering of the LM character
			GlyphChar nextGlyphChar = new GlyphChar(spaceCharIndex, GlyphType.NORMAL_CHAR);
			double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, spaceCharIndex, nextGlyphChar);
			addState(result, nextContext, nextType, nextLanguage, nextGlyphChar, transitionScore + glyphLogProb);
		}
	}

	/**
	 * Add transition states, allowing for the possibility of substitutions or elisions.
	 * 
	 *   1. Next state's glyph is just the rendering of the LM character
	 *   2. Next state's glyph is a substitution of the LM character
	 *   3. Next state's glyph is an elision-decorated version of the LM character
	 *   4. Next state's glyph is elided
	 *   5. Next state's glyph is the LM char, stripped of its accents
	 *   6. Next state's glyph is an elision after a space
	 *   7. Next state's glyph is a doubled version of the LM character
	 * 
	 */
	private void addGlyphStartStates(List<Tuple2<TransitionState, Double>> result, int nextLmChar, int[] nextContext, TransitionStateType nextType, int nextLanguage, double transitionScore) {
		if (!allowGlyphSubstitution)
			addState(result, nextContext, nextType, nextLanguage, new GlyphChar(nextLmChar, GlyphType.NORMAL_CHAR), transitionScore);
		else {
			Set<GlyphChar> potentialNextGlyphChars = new HashSet<GlyphChar>(); 
	
			// 1. Next state's glyph is just the rendering of the LM character
			potentialNextGlyphChars.add(new GlyphChar(nextLmChar, GlyphType.NORMAL_CHAR));
			
			// 2. Next state's glyph is a substitution of the LM character
			if (canBeReplaced.contains(nextLmChar)) {
				for (int nextGlyphCharIndex : lm.get(nextLanguage).getActiveCharacters()) {
					if (validSubstitutionChars.contains(nextGlyphCharIndex)) {
						potentialNextGlyphChars.add(new GlyphChar(nextGlyphCharIndex, GlyphType.NORMAL_CHAR));
					}
				}
			}
			if (nextLmChar == sCharIndex)
				potentialNextGlyphChars.add(new GlyphChar(longsCharIndex, GlyphType.NORMAL_CHAR));
			
			// 3. Next state's glyph is an elision-decorated version of the LM character
			Integer tildeDecorated = addTilde.get(nextLmChar);
			if (tildeDecorated != null) {
				potentialNextGlyphChars.add(new GlyphChar(tildeDecorated, GlyphType.ELISION_TILDE));
			}
			
			// 5. Next state's glyph is the LM char, stripped of its accents
			Integer baseChar = diacriticDisregardMap.get(nextLmChar);
			if (baseChar != null) {
				potentialNextGlyphChars.add(new GlyphChar(baseChar, GlyphType.NORMAL_CHAR));
			}

			// 6. Next state's glyph is an elision after a space --- and the start state is always a "space"
			if (!elideAnything) {
			if (nextType == TransitionStateType.TMPL) {
				if (canBeElided.contains(nextLmChar)) {
					potentialNextGlyphChars.add(new GlyphChar(spaceCharIndex, GlyphType.FIRST_ELIDED));
				}
			}
			}

			// 7. Next state's glyph is a doubled version of the LM character
			if (validDoublableSet.contains(nextLmChar)) {
				potentialNextGlyphChars.add(new GlyphChar(nextLmChar, GlyphType.DOUBLED));
				if (nextLmChar == sCharIndex)
					potentialNextGlyphChars.add(new GlyphChar(longsCharIndex, GlyphType.DOUBLED));
			}
			
			// 8. Elide the character
			if (elideAnything) {
				if (nextType == TransitionStateType.TMPL) {
					if (canBeElided.contains(nextLmChar)) {
						potentialNextGlyphChars.add(new GlyphChar(spaceCharIndex, GlyphType.ELIDED));
					}
				}
			}

			// Create states for all the potential next glyphs
			for (GlyphChar nextGlyphChar : potentialNextGlyphChars) {
				double glyphLogProb = calculateGlyphLogProb(nextType, nextLanguage, nextLmChar, nextGlyphChar);
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
			SingleLanguageModel destLM = lm.get(destLanguage);
			double destLanguagePrior = lm.languagePrior(destLanguage);
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
				totalSpaceProb += getNgramProb(lm.get(language), new int[0], spaceCharIndex) * lm.languagePrior(language);
			double score = Math.log(1.0 - LINE_MRGN_PROB) + Math.log(totalSpaceProb) /*+ Math.log(1.0)*/;
			addNoSubGlyphStartState(result, new int[] { spaceCharIndex }, TransitionStateType.TMPL, -1, score);
		}
		return result;
	}

	private double getNgramProb(SingleLanguageModel slm, int[] context, int c) {
		if (slm != null) {
			return slm.getCharNgramProb(shrinkContext(context, slm), c);
		}
		else {
			// No current language, so sum transition to `c` across all languages
			double totalSpaceProb = 0.0;
			for (int language = 0; language < numLanguages; ++language) {
				SingleLanguageModel languageLM = this.lm.get(language);
				totalSpaceProb += languageLM.getCharNgramProb(shrinkContext(context, languageLM), c) * this.lm.languagePrior(language);
			}
			return totalSpaceProb;
		}
	}

	//	private int[] appendToContext(int[] originalContext, int c, SingleLanguageModels lm) {
	//		return shrinkContext(a.append(originalContext, c), slm);
	//	}

	private double calculateGlyphLogProb(TransitionStateType nextType, int nextLanguage, int nextLmChar, GlyphChar nextGlyphChar) {
		if (nextLanguage < 0) {
			if (this.alwaysSpaceTransitionTypes.contains(nextType) && nextGlyphChar.templateCharIndex == spaceCharIndex)
				return 0.0; // log(1)
			else
				return Double.NEGATIVE_INFINITY; // log(0)
		}
		else {
			double p = (1.0 - noCharSubPrior) * gsm.glyphProb(nextLanguage, nextLmChar, nextGlyphChar);
			double pWithBias = ((nextGlyphChar.glyphType == GlyphType.NORMAL_CHAR && nextGlyphChar.templateCharIndex == nextLmChar) ? noCharSubPrior + p : p);
			return Math.log(pWithBias);
		}
	}

	private int[] shrinkContext(int[] originalContext, SingleLanguageModel slm) {
		int[] newContext = originalContext;
		int maxOrder = slm.getMaxOrder();
		while (newContext.length > maxOrder - 1)
			newContext = ArrayHelper.takeRight(newContext, maxOrder - 1);
		if (slm != null) {
			newContext = slm.shrinkContext(newContext);
		}
		return newContext;
	}
}
