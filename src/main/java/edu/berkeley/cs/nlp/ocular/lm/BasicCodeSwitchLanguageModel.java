package edu.berkeley.cs.nlp.ocular.lm;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.map1;
import indexer.Indexer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.model.LanguageTransitionPriors;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;

/**
 * TODO: Move some of the probability calculations from CodeSwitchTransitionModel to here?
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class BasicCodeSwitchLanguageModel implements CodeSwitchLanguageModel {
	private static final long serialVersionUID = 498752498346537382L;

	private Set<String> languages;

	/**
	 * Map[destinationLanguage -> destinationLM]
	 */
	private Map<String, SingleLanguageModel> subModels;

	/**
	 * Map[destinationLanguage -> destinationLM]
	 */
	private Map<String, Set<String>> wordLists;

	/**
	 * Map[destinationLanguage -> "prior prob of seeing destinationLanguage"]
	 */
	private Map<String, Double> languagePriors;

	/**
	 * Map[destinationLanguage -> Map[fromLanguage, "prior prob of switching from "from" to "destination"]]
	 */
	private Map<String, Map<String, Double>> languageTransitionPriors;

	private Indexer<String> charIndexer;
	private int maxOrder;

	public Set<String> languages() {
		return languages;
	}

	public SingleLanguageModel get(String language) {
		return subModels.get(language);
	}

	public Double languagePrior(String language) {
		return languagePriors.get(language);
	}

	public Double languageTransitionPrior(String fromLanguage, String destinationLanguage) {
		return languageTransitionPriors.get(destinationLanguage).get(fromLanguage);
	}

	public Set<String> wordList(String language) {
		return wordLists.get(language);
	}

	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	public int getMaxOrder() {
		return maxOrder;
	}

	public BasicCodeSwitchLanguageModel(Map<String, Tuple3<SingleLanguageModel, Set<String>, Double>> subModelsAndPriors, Indexer<String> charIndexer, double pKeepSameLanguage, int maxOrder) {
		if (subModelsAndPriors.isEmpty()) throw new IllegalArgumentException("languageModelsAndPriors may not be empty");
		if (pKeepSameLanguage <= 0.0 || pKeepSameLanguage >= 1.0) throw new IllegalArgumentException("pKeepSameLanguage on must be between 0 and 1 (it's " + pKeepSameLanguage + ")");

		// Total prob, for normalizing
		double languagePriorSum = 0.0;
		for (Map.Entry<String, Tuple3<SingleLanguageModel, Set<String>, Double>> lmAndPrior : subModelsAndPriors.entrySet()) {
			double prior = lmAndPrior.getValue()._3;
			if (prior <= 0.0) throw new IllegalArgumentException("prior on " + lmAndPrior.getKey() + " is not positive (it's " + prior + ")");
			languagePriorSum += prior;
		}

		this.languages = new HashSet<String>();
		this.languages.addAll(subModelsAndPriors.keySet());
		
		this.subModels = new HashMap<String, SingleLanguageModel>();
		this.languagePriors = new HashMap<String, Double>();
		this.wordLists = new HashMap<String, Set<String>>();
		for (Map.Entry<String, Tuple3<SingleLanguageModel, Set<String>, Double>> lmAndPrior : subModelsAndPriors.entrySet()) {
			String language = lmAndPrior.getKey();
			this.subModels.put(language, lmAndPrior.getValue()._1);
			this.languagePriors.put(language, lmAndPrior.getValue()._3 / languagePriorSum);
			this.wordLists.put(language, lmAndPrior.getValue()._2);
		}
		

		this.languageTransitionPriors = makeLanguageTransitionPriors(languagePriors, pKeepSameLanguage);

		this.charIndexer = charIndexer;
		this.maxOrder = maxOrder;
	}

	/**
	 * @param languagePriors	Map[destinationLanguage, "prior prob of destinationLanguage"]
	 * @param pKeepSameLanguage	The prior probability of deterministically keeping the same language on a word boundary.
	 * @return Map[destinationLanguage -> Map[fromLanguage, "prob of switching from "from" to "to"]]
	 */
	public static Map<String, Map<String, Double>> makeLanguageTransitionPriors(Map<String, Double> languagePriors, double pKeepSameLanguage) {
		if (languagePriors.isEmpty()) throw new IllegalArgumentException("languagePriors may not be empty");
		if (pKeepSameLanguage <= 0.0 || pKeepSameLanguage >= 1.0) throw new IllegalArgumentException("pKeepSameLanguage must be between 0 and 1, was " + pKeepSameLanguage);

		Set<String> languages = languagePriors.keySet();
		if (languages.size() > 1) {
			double pSwitchLanguages = (1.0 - pKeepSameLanguage) / (languages.size() - 1);

			// Map[destinationLanguage -> Map[fromLanguage, "prob of switching from "from" to "to"]]
			Map<String, Map<String, Double>> result = new HashMap<String, Map<String, Double>>();
			for (String destLanguage : languages) {
				double destPrior = languagePriors.get(destLanguage);
				if (destPrior <= 0.0) throw new IllegalArgumentException("prior on " + destLanguage + " is not positive (it's " + destPrior + ")");

				Map<String, Double> transitionPriors = new HashMap<String, Double>();
				for (String fromLanguage : languages) {
					double transitionProb;
					if (fromLanguage.equals(destLanguage))  // keeping the same language across the transition
						transitionProb = pKeepSameLanguage;
					else
						transitionProb = pSwitchLanguages;
					// prior probability of keeping/switching with the same language and (normalized) prior of switching to destination language
					transitionPriors.put(fromLanguage, transitionProb * destPrior);
				}
				result.put(destLanguage, transitionPriors);
			}

			// Adjust the results map by normalizing the probabilities
			for (String fromLanguage : languages) {
				double transitionPriorSum = 0.0;
				for (String destLanguage : languages) { // Get the total probability for normalization
					transitionPriorSum += result.get(destLanguage).get(fromLanguage);
				}

				for (String destLanguage : languages) { // Normalize all the probabilities so that they sum to 1.0
					double transitionProb = result.get(destLanguage).get(fromLanguage);
					result.get(destLanguage).put(fromLanguage, transitionProb / transitionPriorSum); // normalize the probability and put it back
				}
			}

			return result;
		}
		else {
			// Only one language means no switching ever, so probability of keeping the same language is 1.0
			String language = languages.iterator().next();
			return map1(language, map1(language, 1.0));
		}
	}

	/**
	 * TODO: This is really just here for DenseBigramTransitionModel to use.
	 *       I have *NO IDEA* whether it matters that this doesn't consider: 
	 *         a) the role of spaces in determining language switches (it 
	 *            kind of assumes that every character can be a different 
	 *            language)
	 *         b) languageTransitionPriors (since we can't track the language
	 *            of the context)
	 */
	public double getCharNgramProb(int[] context, int c) {
		//			if(context[context.length-1] == charIndexer.getIndex(Main.SPACE)) { // this is right after a space  
		// assume any language is possible
		double probSum = 0.0;
		for (String language : languages) {
			probSum += subModels.get(language).getCharNgramProb(context, c) * languagePriors.get(language);
		}
		return probSum;
		//			}
		//			else{
		//				
		//			}
	}

}
