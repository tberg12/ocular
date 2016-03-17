package edu.berkeley.cs.nlp.ocular.main;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class NoDocumentsToProcessException extends RuntimeException {
	private static final long serialVersionUID = 1L;

	public NoDocumentsToProcessException() {
		super();
	}

	public NoDocumentsToProcessException(String message) {
		super(message);
	}

}
