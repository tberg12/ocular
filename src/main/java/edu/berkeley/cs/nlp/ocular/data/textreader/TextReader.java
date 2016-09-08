package edu.berkeley.cs.nlp.ocular.data.textreader;

import java.util.List;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public interface TextReader {

	/**
	 * @param line	A line of text, possibly containing diacritics (precomposed, composed, or escaped).
	 * @return	A list of normalized characters.
	 */
	public List<String> readCharacters(String line);
	
}
