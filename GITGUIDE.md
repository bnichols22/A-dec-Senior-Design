# A Beginner's Guide to Git Version Control with GitHub Desktop

## Introduction

(FYI, This was written with AI but I looked it over to make sure there wasn't anything crazy on here)

Welcome to the world of version control! If you've ever saved multiple versions of a file like `project_final.doc`, `project_final_v2.doc`, and `project_REALLY_FINAL.doc`, you already understand the need for a better system. That's where Git and GitHub Desktop come in.

*   **Git** is a free, open-source **Distributed Version Control System (DVCS)**. It tracks changes to your code over time, allowing you to recall specific versions, see who made changes, and easily collaborate with others.
*   **GitHub Desktop** is a graphical user interface (GUI) that makes it easier to use Git without memorizing complex commands in a terminal. It simplifies your workflow while harnessing the full power of Git.

## Core Git Concepts: The "What" and "Why"

Before we click buttons, let's understand the key terms.

1.  **Repository (Repo):** Your project's folder. A repository contains all of your project's files and the entire history of their changes. It's what Git tracks.

2.  **Commit:** A "save point" or "snapshot" of your project at a specific point in time. Each commit has a unique ID, a message describing what was changed, and a reference to the commit that came before it.

3.  **Branch:** An independent line of development. By default, your repo starts with a `main` (or `master`) branch. You can create other branches to work on new features or bug fixes without affecting the stable code on `main`. Think of it like creating a copy of your blueprint to experiment on.

4.  **Merge:** The act of combining the changes from one branch back into another (e.g., merging a finished `feature-branch` into `main`).

5.  **Clone:** Downloading an existing remote repository from a service like GitHub to your local machine.

6.  **Push:** Uploading your local commits to a remote repository (like GitHub.com).

7.  **Pull (or Fetch):** Downloading changes from a remote repository to your local machine to keep it up to date.

---

## Getting Started with GitHub Desktop

### Step 1: Installation and Setup

1.  **Download & Install:** Go to [desktop.github.com](https://desktop.github.com/) and download GitHub Desktop for your operating system (Windows or macOS).
2.  **Authenticate:** Open the application and sign in with your GitHub.com account. If you don't have one, you can create it from the app.
3.  **Configure Identity:** Git needs to know who you are to label your commits. GitHub Desktop will automatically use the name and email from your GitHub account. You can verify this in `Preferences` > `Git`.

### Step 2: Clone This Repository

You should see this repo populate in github desktop, but if you don't you can clone from this HTTP link on desktop through File -> Clone Repository -> URL

https://github.com/bnichols22/A-dec-Senior-Design.git

### Step 3: Making Your First Commit

Now, let's add some work and create a save point.

1.  **Make a Change:** Using your preferred text editor (like VS Code), add a new file to the repository folder GitHub Desktop created. For example, create `index.html` and write some simple HTML.
2.  **View Changes in GitHub Desktop:** Return to GitHub Desktop. You will see your new file listed in the "Changes" panel on the left.
3.  **Stage Changes:** Check the box next to the file(s) you want to include in this snapshot (this is called "staging").
4.  **Write a Commit Message:**
    *   **Summary (Required):** A short, imperative description (e.g., "Create homepage structure").
    *   **Description (Optional):** More detailed notes about what you changed and why (e.g., "Added basic HTML5 boilerplate and a header with navigation links").
    *   *Good commit messages are crucial for understanding your project's history!*
5.  **Commit to `main`:** Click the blue `Commit to main` button at the bottom.

You've made your first commit! The "Changes" panel will be empty, and your commit will now appear in the history graph.

### Step 4: Working with Branches

Let's safely work on a new feature.

1.  **Create a Branch:** Click the `Current Branch` dropdown menu in the top toolbar and select `New Branch`.
2.  **Name Your Branch:** Give it a descriptive name, like `add-styling`. Ensure it's based on `main`. Click `Create Branch`. GitHub Desktop will automatically switch you to this new branch.
3.  **Make Changes:** Now, create a new file, like `styles.css`, and add some CSS rules. Save the file.
4.  **Commit on the Branch:** Go back to GitHub Desktop, stage the new file, write a commit message (e.g., "Add basic styles and color scheme"), and click `Commit to add-styling`.

You've now made a change on a separate branch. Your `main` branch remains untouched and stable.

### Step 5: Publishing to GitHub and Pushing Changes

So far, everything has been local. Now let's put it online.

1.  **Publish Repository:** If you just created the repo locally, you'll see a `Publish repository` button in the top right. Click it.
2.  **Configure Remote:**
    *   Keep the name the same.
    *   Your GitHub account will be selected.
    *   You can add a description.
    *   Choose to keep the repository **Public** (visible to everyone) or **Private** (only visible to you and collaborators).
3.  Click `Publish Repository`. Your local repo is now on GitHub.com!
4.  **Push Future Changes:** After you make more commits locally, you need to "push" them to GitHub. Click the `Push origin` button in the top toolbar to upload your new commits.

### Step 6: Merging a Branch via Pull Request

Your feature is done on the `add-styling` branch. It's time to merge it into `main`.

1.  **Push Your Branch:** Make sure all commits on your `add-styling` branch are pushed to GitHub (`Push origin`).
2.  **Create a Pull Request:** In GitHub Desktop, click the `Current Branch` dropdown and select `Create Pull Request`. This will open your default browser to the Pull Request (PR) page on GitHub.com.
3.  **Open the Pull Request:** On GitHub, confirm the branches are correct (merging `add-styling` *into* `main`). Add a title and description for your PR, then click `Create pull request`.
4.  **Merge the Pull Request:** On the GitHub.com PR page, if there are no conflicts, you will see a `Merge pull request` button. Click it and confirm the merge. This merges the code on GitHub.
5.  **Update Your Local `main`:** Return to GitHub Desktop.
    *   Switch back to your `main` branch using the `Current Branch` dropdown.
    *   Click the `Fetch origin` button (it will change to `Pull origin`) to download the merged changes from GitHub.com to your local `main` branch. Your local `main` is now up to date!

---

## Quick Troubleshooting & FAQ

*   **I made a commit to the wrong branch!**
    *   Don't panic. You can "revert" a commit. Right-click on the commit in the history and select `Revert this Commit`. This creates a new commit that undoes the changes.
*   **GitHub Desktop says I have "merge conflicts"!**
    *   This happens when Git can't automatically combine changes from two branches. GitHub Desktop will guide you through resolving them by letting you choose which changes to keep. It seems scary at first, but it's a normal part of collaboration.
*   **Should I use "Pull" or "Fetch"?**
    *   **Fetch** checks for new changes on GitHub but doesn't automatically apply them to your code. It's a safe way to see what's new.
    *   **Pull** is a `Fetch` followed immediately by a `Merge`. It's the standard way to update your local branch.

## Next Steps

*   **Explore the History Tab:** Click on the `History` tab in GitHub Desktop to see a visual graph of all your commits and branches.
*   **Collaborate:** Add a friend as a collaborator on your GitHub.com repository and practice pushing, pulling, and merging each other's changes.
*   **Learn the Command Line:** As you get more comfortable, learning basic Git commands in the terminal will give you even more power and flexibility.