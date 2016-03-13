package edu.berkeley.cs.nlp.ocular.font;

import java.io.Serializable;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class Font implements Serializable {
	private static final long serialVersionUID = 1L;

	public final Map<String, CharacterTemplate> charTemplates;

	public Font(Map<String, CharacterTemplate> charTemplates) {
		this.charTemplates = charTemplates;
	}
	
	public CharacterTemplate get(String character) {
		return charTemplates.get(character);
	}

}
