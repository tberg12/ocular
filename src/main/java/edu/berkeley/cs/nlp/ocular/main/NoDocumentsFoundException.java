package edu.berkeley.cs.nlp.ocular.main;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class NoDocumentsFoundException extends RuntimeException {
	private static final long serialVersionUID = 1L;

	public NoDocumentsFoundException() {
		super("No documents were found in the given input path(s).");
	}

	public NoDocumentsFoundException(String message) {
		super(message);
	}

}
