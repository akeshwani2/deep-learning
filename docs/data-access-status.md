# Data Access and Governance Status

- **Last reviewed:** 2026-10-22
- **Owners:** Poorav Rawat and Arhaan Keshwani (joint)
- **Selected path:** Mode B — UCI feature-only fallback
- **Sensitive values included:** None

## mPower access tracker

mPower is controlled-access human-participant data. Synapse documentation states
that mHealth access requires a validated user and that each user must separately
satisfy the dataset's Conditions for Use. Depending on the dataset, those
conditions may also require certification, an intended-use statement, a Data Use
Certificate, or institutional/IRB review.

Authoritative references:

- [Synapse user account types](https://docs.synapse.org/synapse-docs/synapse-user-account-types)
- [Sharing settings, permissions, and Conditions for Use](https://docs.synapse.org/synapse-docs/sharing-settings-permissions-and-conditions-for-use)
- [mPower data-access discussion and intended-use process](https://www.synapse.org/Synapse:syn5637528/discussion/threadId=2381)

| Requirement | Repository-verifiable status | Consequence |
|---|---|---|
| Synapse account for every data user | Not documented | Cannot claim authorized access |
| Synapse certification | Not documented | Treat as incomplete |
| Validated user profile for mHealth access | Not documented | Treat as incomplete |
| Intended Data Use submitted/accepted | Not documented | Treat as incomplete |
| Data-specific Conditions for Use accepted | Not documented | Treat as incomplete |
| Institutional/course guidance, if applicable | Not documented | Treat as incomplete |
| Approved raw audio downloaded outside the repository | Not documented | No mPower data may enter the pipeline |

This is an access-status record, not a claim that access was denied. It records
that Mode A cannot pass the project gate using available evidence. No account
names, tokens, application text, approvals, or controlled metadata belong in
this repository.

## Mode A decision deadline and feasibility criteria

The latest Mode A start date is **2026-10-22**. No external delivery date is
recorded, so the governance-completion date is used as the conservative deadline
rather than delaying implementation on an unverified access path.

Mode A requires every criterion below to pass by that deadline:

1. Every team member handling mPower data has authorized access and has accepted
   the applicable Conditions for Use.
2. The approved use covers this non-diagnostic participant-level voice study,
   its storage plan, planned derived artifacts, and intended reporting.
3. Sustained-vowel audio, stable participant identifiers, and usable PD-status
   labels are available and permitted for analysis.
4. A deterministic pilot can join recordings to participant and label fields
   without exposing identifiers in repository artifacts or logs.
5. At least 95% of a representative pilot of 100 recordings (or all recordings
   if fewer are available) decode successfully, with both target classes
   represented. Failures must be recorded using non-identifying reason codes.
6. Institutional or course guidance does not impose an unmet prerequisite.

The available evidence does not satisfy criteria 1–5, and criterion 6 is not
documented. Under the predeclared fallback rule, the project therefore proceeds
in Mode B. A later move to Mode A requires a new decision-log entry that
supersedes DL-002; it must not silently mix datasets or claims.

## Mode B legal and ethical basis

UCI dataset 489 is publicly downloadable and its official catalog entry assigns
the dataset a Creative Commons Attribution 4.0 International license. That
license permits sharing and adaptation with attribution. The project will cite
the official dataset and DOI, identify changes to any redistributed adaptation,
and avoid adding restrictions that conflict with the license.

Mode B does not remove the project's research-only limits. Outputs remain
non-diagnostic, participant-grouped, and subject to the privacy and artifact
handling rules in the requirements and data manifest.
