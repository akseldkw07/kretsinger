#!/usr/bin/env bash
gitrmcache() {
  git rm -r --cached .
  git add .
}
