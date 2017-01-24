package edu.berkeley.cs.nlp.ocular.lm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import tberg.murphy.indexer.Indexer;

/**
 * TODO: Move some of the probability calculations from CodeSwitchTransitionModel to here?
 * 
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BasicCodeSwitchLanguageModel implements CodeSwitchLanguageModel {
	private static final long serialVersionUID = 3298359823L;

	private Indexer<String> langIndexer;

	/**
	 * Map[destinationLanguage -> destinationLM]
	 */
	private List<SingleLanguageModel> subModels;

	/**
	 * Map[destinationLanguage -> "prior prob of seeing destinationLanguage"]
	 */
	private List<Double> languagePriors;

	/**
	 * Map[destinationLanguage -> Map[fromLanguage, "prior prob of switching from "from" to "destination"]]
	 */
	private List<List<Double>> languageTransitionProbs;

	private Indexer<String> charIndexer;
	private double pKeepSameLanguage;

	public Indexer<String> getLanguageIndexer() {
		return langIndexer;
	}

	public SingleLanguageModel get(int language) {
		if (language == -1)
			return null;
		else
			return subModels.get(language);
	}

	public double languagePrior(int language) {
		return languagePriors.get(language);
	}

	public double languageTransitionProb(int fromLanguage, int destinationLanguage) {
		return languageTransitionProbs.get(destinationLanguage).get(fromLanguage);
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	public double getProbKeepSameLanguage() {
		return pKeepSameLanguage;
	}

	public BasicCodeSwitchLanguageModel(List<Tuple2<SingleLanguageModel, Double>> subModelsAndPriors, Indexer<String> charIndexer, Indexer<String> langIndexer, double pKeepSameLanguage) {
		if (subModelsAndPriors.isEmpty()) throw new IllegalArgumentException("languageModelsAndPriors may not be empty");
		if (pKeepSameLanguage <= 0.0 || pKeepSameLanguage > 1.0) throw new IllegalArgumentException("pKeepSameLanguage must be between 0 and 1, was " + pKeepSameLanguage);

		// Total prob, for normalizing
		double languagePriorSum = 0.0;
		for (int langIndex = 0; langIndex < langIndexer.size(); ++langIndex) {
			Tuple2<SingleLanguageModel, Double> lmAndPrior = subModelsAndPriors.get(langIndex);
			double prior = lmAndPrior._2;
			if (prior <= 0.0) throw new IllegalArgumentException("prior on " + langIndexer.getObject(langIndex) + " is not positive (it's " + prior + ")");
			languagePriorSum += prior;
		}

		this.subModels = new ArrayList<SingleLanguageModel>();
		this.languagePriors = new ArrayList<Double>();
		for (Tuple2<SingleLanguageModel, Double> lmAndPrior : subModelsAndPriors) {
			this.subModels.add(lmAndPrior._1);
			this.languagePriors.add(lmAndPrior._2 / languagePriorSum);
		}

		this.languageTransitionProbs = makeLanguageTransitionProbs(this.languagePriors, pKeepSameLanguage, langIndexer);

		this.charIndexer = charIndexer;
		this.langIndexer = langIndexer;
		this.pKeepSameLanguage = pKeepSameLanguage;
	}

	/**
	 * @param languagePriors	Map[destinationLanguage, "prior prob of destinationLanguage"]
	 * @param pKeepSameLanguage	The prior probability of deterministically keeping the same language on a word boundary.
	 * @return Map[destinationLanguage -> Map[fromLanguage, "prob of switching from "from" to "to"]]
	 */
	public static List<List<Double>> makeLanguageTransitionProbs(List<Double> languagePriors, double pKeepSameLanguage, Indexer<String> langIndexer) {
		if (languagePriors.isEmpty()) throw new IllegalArgumentException("languagePriors may not be empty");
		if (pKeepSameLanguage <= 0.0 || pKeepSameLanguage > 1.0) throw new IllegalArgumentException("pKeepSameLanguage must be between 0 and 1, was " + pKeepSameLanguage);

		int numLanguages = langIndexer.size();
		if (numLanguages > 1) {
			double pSwitchLanguages = (1.0 - pKeepSameLanguage) / (numLanguages - 1);

			// Map[destinationLanguage -> Map[fromLanguage, "prob of switching from "from" to "to"]]
			List<List<Double>> result = new ArrayList<List<Double>>();
			for (int destLanguage = 0; destLanguage < numLanguages; ++destLanguage) {
				double destPrior = languagePriors.get(destLanguage);
				if (destPrior <= 0.0) throw new IllegalArgumentException("prior on " + langIndexer.getObject(destLanguage) + " is not positive (it's " + destPrior + ")");

				List<Double> transitionPriors = new ArrayList<Double>();
				for (int fromLanguage = 0; fromLanguage < numLanguages; ++fromLanguage) {
					double transitionProb;
					if (fromLanguage == destLanguage)  // keeping the same language across the transition
						transitionProb = pKeepSameLanguage;
					else
						transitionProb = pSwitchLanguages;
					// prior probability of keeping/switching with the same language and (normalized) prior of switching to destination language
					transitionPriors.add(transitionProb * destPrior);
				}
				result.add(transitionPriors);
			}

			// Adjust the results map by normalizing the probabilities
			for (int fromLanguage = 0; fromLanguage < numLanguages; ++fromLanguage) {
				double transitionPriorSum = 0.0;
				for (List<Double> transitionPriors : result) { // Get the total probability for normalization
					transitionPriorSum += transitionPriors.get(fromLanguage);
				}

				for (List<Double> transitionPriors : result) { // Normalize all the probabilities so that they sum to 1.0
					double transitionProb = transitionPriors.get(fromLanguage);
					transitionPriors.set(fromLanguage, transitionProb / transitionPriorSum); // normalize the probability and put it back
				}
			}

			return result;
		}
		else {
			// Only one language means no switching ever, so probability of keeping the same language is 1.0
			return Collections.singletonList(Collections.singletonList(1.0));
		}
	}

	/**
	 * TODO: This is really just here for DenseBigramTransitionModel to use.
	 *       I have *NO IDEA* whether it matters that this doesn't consider: 
	 *         a) the role of spaces in determining language switches (it 
	 *            kind of assumes that every character can be a different 
	 *            language)
	 *         b) languageTransitionProb (since we can't track the language
	 *            of the context)
	 */
	public double getCharNgramProb(int[] context, int c) {
		//			if(context[context.length-1] == charIndexer.getIndex(Main.SPACE)) { // this is right after a space  
		// assume any language is possible
		double probSum = 0.0;
		for (int language = 0; language < this.langIndexer.size(); ++language) {
			probSum += subModels.get(language).getCharNgramProb(context, c) * languagePriors.get(language);
		}
		return probSum;
		//			}
		//			else{
		//				
		//			}
	}

}
