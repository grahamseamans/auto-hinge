"""
Profile Classifier Tool

A desktop GUI application for manually labeling screenshots with YES/NO classifications,
training machine learning models, and automating profile rejection.

See README.md for detailed specification and usage instructions.
"""

from gui import GUI


def main():
    """Main entry point for the Profile Classifier Tool"""
    gui = GUI()
    gui.create_gui()


if __name__ == "__main__":
    main()
