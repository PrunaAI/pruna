# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from collections import Counter
from pathlib import Path

import github
from github import Github


def pattern_to_regex(pattern):
    """Turn a CODEOWNERS glob ``pattern`` into a regex for matching file paths."""
    if pattern.startswith("/"):
        start_anchor = True
        pattern = re.escape(pattern[1:])
    else:
        start_anchor = False
        pattern = re.escape(pattern)
    pattern = pattern.replace(r"\*", "[^/]*")
    if start_anchor:
        pattern = r"^\/?" + pattern  # Allow an optional leading slash after the start of the string
    return pattern


def get_file_owners(file_path, codeowners_lines):
    """Return owner logins for ``file_path`` using CODEOWNERS rules (last match wins)."""
    for line in reversed(codeowners_lines):
        line = line.split('#')[0].strip()
        if not line:
            continue

        parts = line.split()
        pattern = parts[0]
        # Can be empty, e.g. for dummy files with explicitly no owner
        owners = [owner.removeprefix("@") for owner in parts[1:]]

        file_regex = pattern_to_regex(pattern)
        if re.search(file_regex, file_path) is not None:
            return owners  # It can be empty
    return []


def get_dispatch_owners(codeowners_lines):
    """Return fallback owners from the catch-all ``*`` CODEOWNERS rule."""
    for line in codeowners_lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = line.split()
        if parts[0] == "*":
            return [owner.removeprefix("@") for owner in parts[1:]]

    return []


def main():
    """Load the PR event, skip if reviews exist or reviewers are already requested, then request recent owners."""
    script_dir = Path(__file__).parent.absolute()
    with open(script_dir / "codeowners_assignment") as f:
        codeowners_lines = f.readlines()

    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo("PrunaAI/pruna")
    with open(os.environ['GITHUB_EVENT_PATH']) as f:
        event = json.load(f)

    pr_number = event['pull_request']['number']
    pr = repo.get_pull(pr_number)
    pr_author = pr.user.login

    # Skipping exceptions
    existing_reviews = list(pr.get_reviews())
    if existing_reviews:
        return

    users_requested, teams_requested = pr.get_review_requests()
    users_requested = list(users_requested)
    if users_requested:
        return

    # Counting recent owner matches
    latest_owner_matches = Counter()
    for file in pr.get_files():
        owners = set(get_file_owners(file.filename, codeowners_lines))
        owners.discard(pr_author)
        if not owners:
            continue

        commits = repo.get_commits(path=file.filename)
        for commit in commits:
            if commit.author is None:
                continue

            login = commit.author.login
            if login in owners:
                latest_owner_matches[login] += file.changes
                break

    top_owners = [owner for owner, _ in latest_owner_matches.most_common(2)]

    if not top_owners:
        top_owners = [
            owner for owner in get_dispatch_owners(codeowners_lines)
            if owner != pr_author
        ]
    try:
        pr.create_review_request(top_owners)
    except github.GithubException as e:
        raise e


if __name__ == "__main__":
    main()
