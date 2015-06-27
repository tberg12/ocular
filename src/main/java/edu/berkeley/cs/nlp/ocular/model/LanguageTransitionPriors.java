package edu.berkeley.cs.nlp.ocular.model;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.map1;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class LanguageTransitionPriors {

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
}
