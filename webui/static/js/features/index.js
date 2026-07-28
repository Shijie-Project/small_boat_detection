// Front-end feature registry, keyed by the name the backend reports.
// Adding a tab: a module here + an entry below + the matching webui/features module.

import test from "./test.js";
import train from "./train.js";

export const FEATURE_UI = { train, test };
